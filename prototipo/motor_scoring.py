"""
Motor de scoring del prototipo de bot de cobranza.

Recibe cédula y mes, consulta la tabla de scores pre-calculada y devuelve la
decisión en JSON: score, nivel de riesgo, acción sugerida y respuesta para el
chatbot.

La tabla de scores está indexada por `id_anonimo` (SHA-256 con salt fijo), no por
cédula: el motor hashea la cédula recibida y busca por el hash, de modo que la
tabla en reposo no contiene ningún dato personal identificable. La cédula solo
existe en memoria durante la consulta.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_SCORES = RAIZ / "outputs" / "scores_clientemes.csv"

# Cortes de riesgo sobre el score (probabilidad de CUMPLIMIENTO).
# Score bajo = poco probable que pague = riesgo alto.
CORTE_ALTO = 0.35
CORTE_BAJO = 0.65

# --------------------------------------------------------------------------- #
# Guion de respuestas por nivel de riesgo
# --------------------------------------------------------------------------- #
GUION = {
    "alto": {
        "accion_sugerida": "Derivar a asesor humano y ofrecer facilidades de pago",
        "prioridad_gestion": 1,
        "canal_recomendado": "llamada",
        "respuesta_chatbot": (
            "Entendemos que este mes puede ser complicado. Tenemos alternativas "
            "que se ajustan a tu situación: podemos revisar juntos un plan de pago "
            "o derivarte con un asesor. ¿Te gustaría que te contactemos?"
        ),
    },
    "medio": {
        "accion_sugerida": "Reforzar el compromiso con recordatorio y confirmación de fecha",
        "prioridad_gestion": 2,
        "canal_recomendado": "whatsapp",
        "respuesta_chatbot": (
            "Gracias por comunicarte. Para dejar tu compromiso registrado, "
            "confírmanos la fecha en la que realizarás el pago y te enviaremos "
            "un recordatorio ese día."
        ),
    },
    "bajo": {
        "accion_sugerida": "Autogestión: enviar medios de pago y no saturar con contactos",
        "prioridad_gestion": 3,
        "canal_recomendado": "autoservicio",
        "respuesta_chatbot": (
            "Te compartimos los medios de pago disponibles para que completes tu "
            "operación cuando prefieras. Si necesitas algo más, escríbenos."
        ),
    },
}


class ErrorPrototipo(Exception):
    """Entrada inválida o cliente-mes no encontrado."""


@dataclass
class Decision:
    cedula: str | None
    mes: str
    encontrado: bool
    score: float | None
    nivel_riesgo: str | None
    accion_sugerida: str | None
    prioridad_gestion: int | None
    canal_recomendado: str | None
    respuesta_chatbot: str
    id_anonimo: str | None = None
    modelo: str | None = None
    advertencia: str | None = None

    def to_json(self, indent: int = 2, enmascarar: bool = False) -> str:
        """`enmascarar=True` oculta la cédula: úsalo al mostrar salidas en
        documentos o capturas de pantalla."""
        d = asdict(self)
        if enmascarar and d["cedula"]:
            d["cedula"] = enmascarar_cedula(d["cedula"])
        return json.dumps(d, ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------- #
# Normalización y anonimización — mismas reglas que el pipeline (notebook 01)
# --------------------------------------------------------------------------- #
def normalizar_cedula(valor: str) -> str:
    """Regla 1 del pipeline: solo dígitos, rellenada a 10 con ceros."""
    limpia = re.sub(r"\D", "", str(valor).strip())
    if not limpia:
        raise ErrorPrototipo(f"Cédula sin dígitos: {valor!r}")
    if len(limpia) > 10:
        raise ErrorPrototipo(f"Cédula de más de 10 dígitos: {valor!r}")
    return limpia.zfill(10)


def normalizar_mes(valor: str) -> str:
    """Regla 2 del pipeline: clave temporal YYYYMM."""
    limpio = re.sub(r"\D", "", str(valor).strip())
    if len(limpio) != 6:
        raise ErrorPrototipo(f"Mes debe tener formato YYYYMM: {valor!r}")
    return limpio


def enmascarar_cedula(cedula: str) -> str:
    """0102345678 -> 01******78. Para mostrar en pantalla sin exponer el documento."""
    c = str(cedula)
    return c if len(c) < 5 else c[:2] + "*" * (len(c) - 4) + c[-2:]


def obtener_salt() -> str:
    """El mismo salt del pipeline. Sin él los hashes no coinciden."""
    salt = os.getenv("CAPSTONE_SALT")
    if salt:
        return salt
    ruta = RAIZ / ".salt"
    if not ruta.exists():
        raise ErrorPrototipo(
            "No se encontró el salt. Definir CAPSTONE_SALT o conservar el archivo .salt "
            "generado por el notebook 01: sin él, el hash de la cédula no coincide con "
            "el de la tabla de scores."
        )
    return ruta.read_text(encoding="utf-8").strip()


def hashear(cedula: str, salt: str) -> str:
    return hashlib.sha256((salt + cedula).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
def clasificar_riesgo(score: float) -> str:
    """Score = probabilidad de cumplimiento. Menor score, mayor riesgo."""
    if score < CORTE_ALTO:
        return "alto"
    if score <= CORTE_BAJO:
        return "medio"
    return "bajo"


class MotorScoring:
    def __init__(self, ruta_scores: Path | str = RUTA_SCORES):
        self.ruta = Path(ruta_scores)
        if not self.ruta.exists():
            raise ErrorPrototipo(
                f"No existe la tabla de scores en {self.ruta}. "
                "Generarla con el notebook 05."
            )
        tabla = pd.read_csv(self.ruta, dtype={"id_anonimo": str, "mes": str})
        self.scores = tabla.set_index(["id_anonimo", "mes"])["score"].to_dict()
        self.modelo = tabla["modelo"].iloc[0] if "modelo" in tabla.columns else None
        self.advertencia = (
            tabla["advertencia"].iloc[0] if "advertencia" in tabla.columns else None
        )
        self.salt = obtener_salt()

    def consultar(self, cedula: str, mes: str) -> Decision:
        ced = normalizar_cedula(cedula)
        m = normalizar_mes(mes)
        return self._decidir(ced, m, hashear(ced, self.salt))

    def consultar_por_id(self, id_anonimo: str, mes: str) -> Decision:
        """Consulta por seudonimo. Sirve cuando no se dispone de los datos crudos,
        que es el caso de quien recibe el proyecto sin la carpeta bases/."""
        return self._decidir(None, normalizar_mes(mes), str(id_anonimo).strip())

    def _decidir(self, ced, m, id_anonimo) -> Decision:
        clave = (id_anonimo, m)

        if clave not in self.scores:
            return Decision(
                cedula=ced, mes=m, id_anonimo=id_anonimo, encontrado=False,
                score=None, nivel_riesgo=None,
                accion_sugerida=None, prioridad_gestion=None, canal_recomendado=None,
                respuesta_chatbot=(
                    "No encontramos una gestión activa para este documento en el "
                    "período consultado. Verifica el número de cédula o comunícate "
                    "con un asesor."
                ),
            )

        score = float(self.scores[clave])
        nivel = clasificar_riesgo(score)
        guion = GUION[nivel]
        return Decision(
            cedula=ced, mes=m, id_anonimo=id_anonimo, encontrado=True,
            score=round(score, 4), nivel_riesgo=nivel,
            modelo=self.modelo, advertencia=self.advertencia,
            **{k: guion[k] for k in
               ("accion_sugerida", "prioridad_gestion", "canal_recomendado",
                "respuesta_chatbot")},
        )


def consultar(cedula: str, mes: str, ruta_scores: Path | str = RUTA_SCORES) -> str:
    """Atajo de un solo uso: devuelve el JSON de la decisión."""
    return MotorScoring(ruta_scores).consultar(cedula, mes).to_json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("uso: python motor_scoring.py <cedula> <mes YYYYMM>")
        raise SystemExit(1)
    print(consultar(sys.argv[1], sys.argv[2]))

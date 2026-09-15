# Modelo predictivo de cumplimiento de compromisos de pago

Proyecto de titulación · **Maestría en Inteligencia Artificial Aplicada (UDLA)**
Datos operados por **Amapola Technologies Corp.** sobre un chatbot de cobranza bancaria.

## Objetivo

Predecir si un cliente en mora **cumplirá el compromiso de pago** adquirido en el mes, a partir
de la **secuencia de intenciones conversacionales** que activó en el chatbot, combinada con su
situación de cartera. El modelo es un **XGBoost** binario entrenado sobre un dataset con grano
*una fila por cliente-mes*.

- **Variable objetivo:** `cumplimiento_conciliado` — 1 si el banco confirmó el pago en ese mes.
- **Etiqueta de comparación:** `cumplimiento_ceroteo` — 1 si `dias_mora_cierre = 0`; proxy interno,
  se usa solo para contrastar resultados, no como target de entrenamiento.
- **Ventana del estudio:** julio 2025 – abril 2026 (10 meses).

## Estructura

```
capstone_mia/
├── bases/          CSV exportados de BigQuery (datos crudos, NO versionados)
│   ├── registros_gestion.csv
│   └── registros_pagos.csv
├── notebooks/      Un notebook por fase
│   └── 01_pipeline_datos.ipynb
├── modelos/        Modelo entrenado (.pkl)
├── outputs/        Dataset final, gráficos y métricas
└── README.md
```

## Fuentes de datos

### Fuente 1 — BigQuery (CSV en `bases/`)

| Archivo | Grano | Campos |
|---|---|---|
| `registros_gestion.csv` | cliente-mes | `cedula`, `mes`, `flujo`, `fecha_carpeta_gestion`, `dias_mora_inicial`, `tipo_operacion`, `bucket`, `mes_campania`, `lote_control`, `fecha_cierre_mes`, `dias_mora_cierre` |
| `registros_pagos.csv` | cliente-mes | `cedula`, `mes`, `flujo`, `dias_mora_cierre`, `fecha_pago`, `monto_recuperado`, `cumplimiento_conciliado`, `cumplimiento_ceroteo` |

Del registro de gestión se toma el **primer** registro del mes para las variables predictoras y el
**último** para el cierre. Los exports actuales ya vienen colapsados a cliente-mes (26.199 filas,
12.090 clientes, sin duplicados por `cedula`+`mes`); la agregación del pipeline es idempotente y
sigue siendo válida si en el futuro el export vuelve al grano cliente-día.

**Filtro de universo:** `flujo = COBRANZA` según el CSV de **gestión** (21.723 de 26.199
cliente-mes; el resto es CONVENIO). El `flujo` del CSV de pagos discrepa en 814 cliente-mes
(564 COBRANZA→CONVENIO y 250 al revés): se respeta el de gestión, porque define el universo de
clientes gestionados, que es el marco de la predicción. Pagos solo aporta las etiquetas.

### Fuente 2 — MySQL local (esquema verificado el 2026-08-09)

Las cuatro tablas están en el esquema `mia_capstone`:

| Tabla | Filas | Aporta |
|---|---|---|
| `prod_clients_intents` | 65.944 | `id_cliente`, `id_intent`, `created_at` |
| `prod_intents` | 117 ids / **60 nombres** distintos | `intent` |
| `prod_db` | 18.635 clientes (ids 1–42.512) | **`cedula`** y **`respond`** |
| `prod_compromisos` | 2.812 | `cedula`, `fecha` → plazo del compromiso |

`prod_clients_intents` **no tiene columna `cedula` ni `respond`**: ambas viven en `prod_db`, por lo
que `respond` describe al **cliente**, no al evento conversacional. `prod_db` es la tabla que
traduce `id_cliente` → `cedula`; con la carga histórica los eventos sin traducción bajaron al
**18,7 %** (antes, con el snapshot de `robocob`, eran 56,6 %).

Credenciales por variable de entorno (nunca en el notebook):

```powershell
$env:MYSQL_HOST = "127.0.0.1"; $env:MYSQL_PORT = "3306"
$env:MYSQL_USER = "root"; $env:MYSQL_PASSWORD = ""; $env:MYSQL_DATABASE = "mia_capstone"
```

### Cobertura conversacional real

Medida contra el universo de 21.642 cliente-mes / 10.452 clientes:

| Escenario | Eventos | Cédulas que cruzan | Cliente-mes con conversación |
|---|---|---|---|
| Sin filtro `respond` (usado) | 29.631 | 4.595 (44,0 %) | **26,5 %** |
| Con `respond = 2` | 1.629 | 227 (2,2 %) | 1,5 % |

`respond` no define el universo de cobranza —eso lo hace `flujo = COBRANZA`— y aplicarlo dejaría
demasiado poco para entrenar, así que `FILTRAR_RESPOND = False`. La celda 3.4 recalcula ambos
escenarios en cada ejecución para dejar constancia del criterio.

## Reglas críticas de cruce

1. **Cédula:** normalizada a texto de 10 dígitos con `zfill(10)` en las tres fuentes antes de
   cualquier join (previa limpieza de caracteres no numéricos y del sufijo `.0` de lecturas float).
   Además se **valida**: provincia 01–24 o 30, tercer dígito 0–5 y dígito verificador módulo 10.
   Los 47 documentos que no pasan la validación (39 RUC de entidad, 6 con dígito verificador
   incorrecto y 2 pasaportes) se **excluyen**: son 81 cliente-mes, 0,4 % del universo.
2. **Mes:** clave temporal `YYYYMM` en las tres fuentes, siempre construida en pandas a partir de
   la fecha de origen.
3. **Desbalance:** el target principal es minoritario; se maneja en el modelado con
   `scale_pos_weight` y validación con Stratified K-Fold. El valor medido sobre el dataset final es
   **4,7 : 1** (`scale_pos_weight ≈ 4,66`, tasa de positivos 17,7 %), no 1:18.

   El 1:18 de la propuesta inicial no se reproduce en los datos exportados: sobre el total son
   4.568 positivos contra 21.631 negativos, y filtrando COBRANZA, 3.696 contra 17.713. Usar 18
   cuando el ratio real es 4,7 sobre-pondera la clase positiva casi cuatro veces: dispara el
   recall, hunde la precisión y descalibra las probabilidades, que es justo lo que se necesita
   bien calibrado para priorizar gestión. En la memoria conviene reportar el ratio medido sobre el
   dataset final.

## Variables predictoras (10)

| # | Variable | Origen | Descripción |
|---|---|---|---|
| 1 | `total_intenciones` | MySQL | Intenciones activadas en el mes |
| 2 | `intenciones_distintas` | MySQL | Intenciones únicas en el mes |
| 3 | `activo_compromiso` | MySQL | 1 si activó intención de compromiso de pago |
| 4 | `dias_compromiso` | MySQL | Plazo elegido, de `robocob.prod_compromisos` como `fecha - created_at` |
| 5 | `activo_ya_pague` | MySQL | 1 si activó la intención "ya pagué" |
| 6 | `ultima_intencion_antes_compromiso` | MySQL | Intención inmediatamente previa al primer compromiso del mes |
| 7 | `hora_primera_interaccion` | MySQL | Hora (0-23) de la primera interacción del mes |
| 8 | `dia_semana_primera_interaccion` | MySQL | Día de la semana (0 = lunes … 6 = domingo) |
| 9 | `dias_mora_inicial` | BigQuery | Días de mora al inicio del mes |
| 10 | `meses_consecutivos_mora` | Derivada | Meses calendario consecutivos con mora al inicio, terminando en el mes actual |

`meses_consecutivos_mora` se calcula sobre el **historial completo de gestión**, antes de filtrar
COBRANZA: la mora es un hecho del cliente y un mes gestionado en CONVENIO no debe cortar la racha.

Se conservan además como contexto de cartera `tipo_operacion`, `bucket`, `mes_campania`,
`lote_control` y `tiene_conversacion` (marca de cliente-mes con o sin registro en el chatbot).

**Excluidos por fuga de información** (se conocen solo después del resultado del mes):
`dias_mora_cierre`, `fecha_pago`, `monto_recuperado`, `fecha_cierre_mes`, `fecha_carpeta_gestion`.
Pueden reincorporarse para análisis descriptivo con `INCLUIR_CAMPOS_POST_RESULTADO = True`.

## Protección de datos

El dataset exportado está **anonimizado**:

- `cedula` se reemplaza por `id_anonimo` = SHA-256 de `salt + cedula`, con salt fijo leído de la
  variable de entorno `CAPSTONE_SALT` o del archivo `.salt` (generado una sola vez y **excluido del
  repositorio**). Sin ese salt los hashes no son reproducibles entre sesiones.
- Se eliminan todos los campos de identificación directa (nombre, tarjeta, dirección, correo,
  celular, teléfono, número de cuenta, identificadores de usuario).
- Una auditoría automática recorre los valores del dataset buscando patrones de dato personal
  (correo, cédula de 10 dígitos, tarjeta, celular) y **aborta la exportación** si encuentra alguno.

Los datos crudos de `bases/` contienen cédulas y no deben versionarse ni compartirse.

## Fases

| Notebook | Estado | Contenido |
|---|---|---|
| `01_pipeline_datos.ipynb` | Listo | Carga, normalización, agregación cliente-mes, cruce, validación, anonimización y export |
| `02_eda.ipynb` | Listo | Perfilado, target, tasas por variable con IC de Wilson, correlaciones, información mutua, sesgo de selección y definición de la población de modelado |
| `03_modelo_xgboost.ipynb` | Listo | Split temporal, dos criterios de CV, `scale_pos_weight`, líneas base, PR-AUC, calibración, curva de lift |
| `04_interpretabilidad.ipynb` | Listo | SHAP sobre XGBoost, coeficientes de la logística, concordancia entre modelos, ablación por bloques |
| `05_prototipo_bot.ipynb` | Listo | Tabla de scores, validación de los cortes de riesgo y demostración del motor de decisión |
| `06_resultados.ipynb` | Listo | Tablas exportables para la memoria (ablación, comparativa, significancia) |
| `07_figuras_documento.ipynb` | Listo | Figuras 8–13 del documento como PNG a 220 dpi |

Los cinco notebooks están **ejecutados con sus outputs guardados**: al abrirlos se ven las tablas y
figuras sin necesidad de volver a correrlos.

## Salidas de la fase 1

- `outputs/dataset_clientemes_v1.csv` — dataset analítico anonimizado: **21.642 cliente-mes ×
  19 columnas**, 10.452 clientes, 10 meses (202507–202604).
- `outputs/diccionario_dataset_v1.csv` — diccionario de datos (tipo, nulos, únicos, rol).

Cifras del dataset final:

| Métrica | Valor |
|---|---|
| `cumplimiento_conciliado` | 3.827 positivos / 17.815 negativos · 17,68 % · `scale_pos_weight` ≈ 4,66 |
| `cumplimiento_ceroteo` | 5.264 / 16.378 · 24,32 % · `scale_pos_weight` ≈ 3,11 |
| Cliente-mes con conversación | 26,5 % |
| Cliente-mes con compromiso | 2.848 |
| Cobertura de `dias_compromiso` | 5,1 % |

## Salidas de la fase 2

- `outputs/dataset_modelado_v1.csv` — población de modelado: **18.580 cliente-mes**, 9.906
  clientes, **20,49 %** de positivos, `scale_pos_weight` ≈ **3,880**, cobertura de chatbot 29,6 %.
- `outputs/figuras/` — 17 figuras PNG.

### Hallazgos del EDA

**El proxy no sustituye al target.** Acuerdo del 85,7 % y kappa de Cohen 0,571 entre
`cumplimiento_ceroteo` y `cumplimiento_conciliado`. El proxy marca 2.268 cumplimientos (10,5 %)
sin pago conciliado: la mora llega a cero también por refinanciación, castigo o ajuste contable.
Confirma la decisión de usar el target conciliado.

**La conversación se asocia a 2,5× más cumplimiento** (31,6 % contra 12,7 %), y la brecha
sobrevive al controlar por bucket de mora: 11,1 puntos porcentuales en promedio dentro de cada
bucket. Es **asociación, no efecto causal** — el chatbot registra conversación solo cuando el
cliente contesta, y quien contesta ya tenía más intención de pagar. Debe reportarse como tal.

**Ranking univariado.** Por información mutua manda `dias_mora_inicial` (0,079), seguida de
`meses_consecutivos_mora` (0,038). Por correlación de Spearman con el target, el bloque
conversacional encabeza: `activo_ya_pague` (0,238), `intenciones_distintas` (0,226),
`total_intenciones` (0,220).

### Dos exclusiones que definen la población de modelado

**1 · Celdas mes × lote con el target ausente (1.777 registros).** El análisis se hace a nivel
mes × lote, no por lote: un mismo lote puede estar bien conciliado un mes y no el siguiente, y
agregarlo esconde el problema.

Una celda con conciliación exactamente cero puede ser real o puede ser un target que no llegó.
El criterio es un **test binomial exacto** contra una referencia de mora comparable (±20 %):

| mes × lote | n | ceroteo | mora | tasa de referencia | P(cero por azar) |
|---|---|---|---|---|---|
| 202603 × `53_ad` | 1.205 | 10,8 % | 125 d | 2,3 % | ~10⁻²⁹ |
| 202604 × `57` | 572 | 12,4 % | 49 d | 10,8 % | ~10⁻²⁹ |

Un cero exacto sobre 1.205 y 572 registros no ocurre por azar, así que **no es comportamiento**.
El flag además es internamente consistente: en todo el dataset `cumplimiento_conciliado = 0` tiene
0 % de `fecha_pago` y de `monto_recuperado`, de modo que no hay pagos que el flag esté ignorando —
lo que falta es el registro entero.

**La causa no se puede identificar con estos datos.** Se probó la hipótesis de que fueran lotes
cargados tarde, con poca ventana de gestión antes del cierre, y no se sostiene: los días
disponibles desde la carga hasta fin de mes correlacionan **−0,165** con el cumplimiento (al revés
de lo esperado) y una de las dos celdas tenía el mes entero por delante. La ventana estaba
confundida con la composición de mora de cada lote. Puede ser conciliación pendiente, un canal de
pago distinto o un proceso que no corrió: **es una pregunta para Amapola, no para el dataset.**

**2 · `dias_mora_inicial = 0` (1.285 registros).** Cuentas al día (bucket B0 y sin bucket): no hay
compromiso de pago que cumplir. Reciben chatbot en 1,6 % de los casos frente al 28 % del resto.

Ambas exclusiones están detrás de banderas reversibles (`EXCLUIR_TARGET_AUSENTE`,
`EXCLUIR_SIN_MORA_INICIAL`) en la sección 6.3 del notebook. Al aplicarlas la serie mensual se
vuelve coherente: 202603 pasa de 11,1 % a 16,8 % y 202604 de 18,9 % a 27,2 %.

### Recomendaciones para la fase 3

- **Split temporal, no aleatorio.** La cobertura del chatbot sube de 5 % a 30 % en la ventana y la
  tasa de cumplimiento se mueve entre 12 % y 41 %. Sugerido: entrenar 202507–202601, validar
  202602–202604.
- **StratifiedGroupKFold agrupando por `id_anonimo`**: un cliente aparece en varios meses y no
  puede quedar a la vez en train y test.
- **No usar `lote_control` ni `mes` como features**: cada lote cubre 1–2 meses, así que es un proxy
  del mes y con split temporal los lotes de prueba son categorías nunca vistas.
- **No imputar nulos**: `dias_compromiso` y `hora`/`dia_semana` son nulos estructurales y XGBoost
  aprende la dirección del faltante.
- **PR-AUC como métrica principal**, más recall y precisión en el umbral operativo, y reporte de
  calibración: el valor del modelo está en priorizar gestión, y para eso la probabilidad tiene que
  significar algo.

## Salidas de la fase 3

- `modelos/xgboost_cumplimiento_v1.pkl` — modelo operativo con sus features e hiperparámetros.
- `outputs/metricas_modelo_v1.json` — métricas completas.
- Figuras 18–21: curvas PR/ROC, lift por capacidad, calibración e importancia por ganancia.

**Split temporal:** train 202507–202601 (11.078 filas, 22,68 % positivos) · test 202602–202604
(7.502 filas, 17,25 % positivos). `scale_pos_weight` = 3,408, calculado solo en train.

### Resultados en test (tasa base 0,1725)

Son **8 configuraciones de 3 familias** (dummy de referencia, regresión logística y XGBoost), no 8
algoritmos distintos. Esta tabla es una comparativa de configuraciones, no el análisis de
alternativas técnicas.

| # | Modelo | Familia | PR-AUC | ROC-AUC | Brier | Lift |
|---|---|---|---|---|---|---|
| 1 | Logística completa | LogisticRegression | **0,4543** | 0,7724 | **0,1377** | 2,63 |
| 2 | XGBoost completo · CV temporal | XGBClassifier | 0,4458 | **0,7912** | 0,1616 | 2,58 |
| 3 | Logística operativa | LogisticRegression | 0,4209 | 0,7644 | 0,1421 | 2,44 |
| 4 | XGBoost completo · monótono | XGBClassifier | 0,4067 | 0,7816 | 0,1623 | 2,36 |
| 5 | XGBoost operativo · CV temporal | XGBClassifier | 0,3923 | 0,7751 | 0,1635 | 2,27 |
| 6 | XGBoost operativo · monótono | XGBClassifier | 0,3895 | 0,7737 | 0,1665 | 2,26 |
| 7 | XGBoost operativo · CV aleatoria | XGBClassifier | 0,3782 | 0,7648 | 0,1601 | 2,19 |
| 8 | Tasa base (dummy) | DummyClassifier | 0,1725 | 0,5000 | 0,1457 | 1,00 |

**La línea base lineal supera a XGBoost con los dos conjuntos de variables**, y está mejor calibrada
en ambos. Pero la ventaja **solo es estadísticamente significativa en el conjunto operativo** (ver
más abajo). XGBoost gana en ROC-AUC, la métrica menos informativa con clases desbalanceadas.

Se probaron dos explicaciones:

- **Descartada — relaciones no monótonas.** Se aplicaron restricciones de monotonía con las
  direcciones sacadas del EDA (mora ↓, racha ↓, intenciones ↑). Si los árboles estuvieran ajustando
  curvas que no transfieren, esto habría cerrado la brecha. **Empeoró**: −0,0028 en el operativo y
  −0,0391 en el completo. La forma de las relaciones no es el problema.
- **En pie — poco dato para la capacidad del modelo.** 11.078 filas de entrenamiento para 13
  variables. El boosting necesita volumen para estimar interacciones estables, y encima la tasa
  base cae de 22,7 % en train a 17,3 % en test.

### ¿Es real la ventaja de la línea base?

Bootstrap pareado, 2.000 remuestreos del test evaluando ambos modelos sobre las mismas filas:

| Comparación | Diferencia | IC 95 % | P(gana logística) | Veredicto |
|---|---|---|---|---|
| Logística vs XGBoost, **operativo** | +0,0284 | [+0,0135, +0,0438] | 100 % | **Significativa** |
| Logística vs XGBoost, **completo** | +0,0087 | [−0,0039, +0,0223] | 89,3 % | **Cabe en el ruido** |

La lectura precisa: en el modelo que **sí puede desplegarse** (sin `activo_ya_pague`), la logística
gana de forma significativa. En el modelo completo, que incluye la variable con fuga temporal,
están empatadas.

Además la ventaja no es estable mes a mes: la logística gana en 202602 y 202603, pero en 202604
—el mes más reciente— XGBoost se pone delante en ambos conjuntos. Con tres meses de test no hay
base para declarar una superioridad estable de una familia sobre la otra.

Es un resultado a reportar, no a esconder: la contribución del trabajo son las variables
conversacionales y el pipeline que las construye. Un experimento que descarta una explicación
plausible vale tanto como uno que confirma otra.

### Tres hallazgos metodológicos

**1 · `activo_ya_pague` es fuga temporal.** Contrastando la marca de tiempo de cada intención
contra `fecha_pago`:

| intención | antes del pago | mismo día | después | mediana |
|---|---|---|---|---|
| `ya_pague` | 38,6 % | 33,1 % | 28,3 % | **0 días** |
| `compromiso` | 73,6 % | 7,6 % | 18,8 % | **−6 días** |

`compromiso` precede al pago y es legítimamente predictiva. `ya_pague` es en buena parte una
confirmación de que el pago ya ocurrió. Por eso se entrenan dos modelos: el **completo** documenta
la especificación de las 10 variables y el **operativo** (sin `ya_pague`) es el que puede
reproducirse en producción. La diferencia en XGBoost es **+0,0535** de PR-AUC — exactamente el
desempeño que no existiría al predecir de verdad.

**2 · La CV aleatoria mentía sobre el desempeño futuro.** Seleccionar hiperparámetros con
`StratifiedGroupKFold` prometía PR-AUC 0,5477 y entregó 0,3782 (cae 0,1695). Con CV temporal
expansiva (entrenar con los meses previos, validar con el siguiente) prometía 0,4993 y entregó
0,3923 (cae 0,1070): **mejor desempeño final con dos tercios del optimismo**. El criterio de
validación importa tanto por el modelo que elige como por la honestidad de la estimación.

**3 · Entrenar contra el proxy cuesta poco en ranking.** La correlación de Spearman entre el
ranking del modelo entrenado contra `cumplimiento_ceroteo` y el entrenado contra el target real es
**≈ 0,96**. Para priorizar gestión los dos ordenan casi igual; la diferencia importa si se necesitan
probabilidades interpretables.

### Uso operativo

Gestionando el **20 % de la cartera** mejor rankeado por el modelo operativo se capturan 598 de los
1.294 cumplidores del período: **46,2 % de recall con 39,9 % de precisión**, 2,31 veces la tasa
base. Al 5 % de capacidad la precisión sube a 53,3 %.

**Las probabilidades no están calibradas** y no deben leerse como tales: `scale_pos_weight` las
infla a propósito (media predicha 0,314 contra una tasa real de 0,172). Para rankear no afecta —
el orden es el mismo—, pero reportar probabilidades interpretables exige recalibrar (isotónica o
Platt) sobre un conjunto aparte.

`tiene_conversacion` no se usa en ningún corte del modelo: la información ya está contenida en
`total_intenciones` e `intenciones_distintas`.

## Salidas de la fase 4

Figuras 22–26: importancia SHAP, dependencia y forma del efecto, coeficientes de la logística,
concordancia entre modelos y ablación por bloques.

### El resultado que responde a la hipótesis del trabajo

Ablación por bloques, reentrenando desde cero con cada subconjunto (PR-AUC en test, tasa base
0,1725):

| Conjunto | n variables | Logística | XGBoost |
|---|---|---|---|
| Solo cartera | 5 | 0,3191 | 0,3413 |
| Solo conversacional | 8 | **0,3492** | 0,3436 |
| Ambos | 13 | **0,4209** | 0,3926 |

**El bloque conversacional por sí solo predice mejor que el bloque de cartera** (0,3492 contra
0,3191 en la logística), y combinarlos aporta +0,102 sobre la cartera sola. Es la evidencia más
directa a favor de la hipótesis del trabajo: las secuencias de intenciones del chatbot no son un
complemento marginal de los datos tradicionales de mora, son una fuente de señal comparable.

Matiz necesario: el predictor individual más fuerte sigue siendo `dias_mora_inicial` (40,2 % del
|SHAP| total), y el bloque conversacional concentra el 24,3 %. Lo conversacional no desplaza a la
cartera, la complementa.

### Concordancia entre los dos modelos

Correlación de Spearman entre el orden de importancia por SHAP y por coeficientes: **0,720** —
coinciden en lo esencial (la mora manda) pero discrepan en `bucket`, que la logística infla por
colinealidad con `dias_mora_inicial`.

### Dos coeficientes que no deben leerse literalmente

La sección 3.1 del notebook documenta dos trampas de colinealidad que, copiadas tal cual a la
memoria, afirmarían cosas falsas:

- **`total_intenciones` sale con coeficiente −0,40** cuando su correlación marginal con el target
  es **+0,200**. Causa: tiene correlación de Spearman **0,997** con `intenciones_distintas`. Dos
  regresores casi idénticos se reparten el efecto de forma arbitraria y a menudo con signos
  opuestos.
- **`bucket_B6` aparece con odds ratio 10,5**, el mayor de la tabla, pero B6 tiene una tasa de
  cumplimiento real del **3,9 %**. El bucket es una discretización de `dias_mora_inicial`, que ya
  está en el modelo como continua: el coeficiente es un ajuste residual, no un efecto.

Regla práctica: la **dirección** se lee del análisis marginal del EDA, la **magnitud** de SHAP, y
los coeficientes se tratan como lo que son — efectos condicionados al resto del modelo.

### Nota sobre las figuras

Paleta categórica validada para daltonismo con el validador de la guía de visualización
(peor par all-pairs: ΔE 9,2 deutan / 24,0 visión normal sobre superficie `#fcfcfb`). Sin dobles
ejes; magnitud en un solo tono; correlaciones en divergente azul↔rojo con gris neutro en el cero.
Cada figura va acompañada de su tabla de respaldo en el notebook: ningún valor se lee solo del
gráfico.

## Instalación

El proyecto usa un entorno virtual propio en `.venv`, ya creado y registrado como kernel de
Jupyter con el nombre **Python (capstone_mia)**.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name capstone_mia --display-name "Python (capstone_mia)"
```

## Ejecución

### Qué se puede ejecutar y qué no

Los datos crudos (`bases/`) contienen cédulas sin anonimizar y **no se distribuyen** con el
proyecto. Por ende, la reproducibilidad está dividida en dos niveles:

| Notebook | ¿Corre en una copia entregada? | Requiere |
|---|---|---|
| `01_pipeline_datos` | No | CSV crudos en `bases/` y acceso a MySQL |
| `02` a `07` | **Sí** | Solo los datasets anonimizados de `outputs/` |
| `prototipo/` | **Sí** | Solo `outputs/scores_clientemes.csv` |

Es decir, cualquiera puede reproducir el análisis exploratorio, el modelado, la interpretabilidad,
las tablas, las figuras y el prototipo. Únicamente la construcción del dataset a partir de las
fuentes originales queda restringida, que es el comportamiento correcto para un trabajo con datos
personales.

### Pasos

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name capstone_mia --display-name "Python (capstone_mia)"
```

Abrir Jupyter y seleccionar el kernel **Python (capstone_mia)**:

```bash
.venv\Scripts\jupyter.exe notebook
```

Los notebooks ya vienen ejecutados con sus resultados guardados, de modo que pueden leerse sin
volver a correrlos. Para reejecutarlos, hacerlo en orden del 02 al 07, ya que cada uno consume las
salidas del anterior.

### Prototipo

```bash
.venv\Scripts\python.exe prototipo/demo.py
```

Recorre cuatro casos, uno por nivel de riesgo más un identificador sin gestión, y cierra con la
distribución de la cartera. Avanza con Enter para permitir la narración; con `--auto` corre solo.

Cuando `bases/` está disponible el prototipo consulta por cédula, que es el flujo real del bot.
Cuando no lo está, consulta por seudónimo y devuelve exactamente la misma decisión, dado que la
tabla de scores se indexa por el hash y no por el documento.

### Requisito del salt

El archivo `.salt` tampoco se distribuye, ya que con él los hashes del dataset serían reversibles
por fuerza bruta. Sin el salt no es posible consultar por cédula, pero sí por seudónimo, que es el
modo en que funciona la demostración en una copia entregada.

---

## Pendientes

Nada de esto es código: son decisiones y datos que dependen de terceros.

**1 · Preguntar a Amapola por dos celdas sin conciliación.** ¿Por qué las cargas de marzo 2026 del
lote `53_ad` (1.205 registros) y de abril 2026 del lote `57` (572) no tienen ni un solo pago
conciliado, cuando sí tienen clientes que llegaron a mora cero? Los datos descartan que sea
comportamiento (P ≈ 10⁻²⁹) pero no identifican la causa. Si resulta que el cero es real, revertir
`EXCLUIR_TARGET_AUSENTE` en la sección 6.3 del notebook 02.

**2 · Reformular el OE3 y la sección de alcance.** Declaran XGBoost como modelo principal, pero la
línea base lineal lo supera en PR-AUC de forma estadísticamente significativa. La redacción
defendible no es sustituir un modelo por otro sino reformular el objetivo hacia la comparación:
*"construir y evaluar un modelo predictivo a partir de secuencias conversacionales, contrastando
XGBoost contra una línea base regularizada"*. La hipótesis del trabajo nunca fue sobre el
algoritmo, y esa sí se sostiene.

**3 · Ampliar la ventana temporal si aparecen datos anteriores.** Comprobado el 2026-08-09:
`bases/` solo contiene 202507–202604. MySQL sí tiene historia previa (intenciones desde el
2024-09-25, compromisos desde el 2024-12-24), pero **faltan los CSV de gestión y pagos anteriores a
julio 2025**, que son los que traen el target.

Si aparecen, el pipeline los absorbe sin cambios de código —basta reemplazar los CSV en `bases/` y
reejecutar los cuatro notebooks— y conviene reentrenar para comprobar si XGBoost alcanza a la
logística con más volumen. Es la hipótesis que quedó en pie tras descartar la de monotonía: con
11.078 filas de entrenamiento el boosting no tiene con qué estimar interacciones estables.

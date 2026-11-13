"""Agent analyste de données conversationnel.

L'utilisateur donne un lien vers un fichier CSV de son Google Drive, puis pose
ses questions en langage naturel. L'agent récupère le fichier via le connecteur
Google Drive (outils du connecteur), puis l'analyse avec des outils fixes :
chargement, description des colonnes, filtrage, agrégats par groupe. Aucune
exécution de code arbitraire : pas de sandbox, pas d'outil d'exécution.
"""

import base64
import csv
import io
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from mistralai import agents

MAX_PREVIEW_ROWS = 3
MAX_FILTER_RETURNED_ROWS = 100
MAX_GROUP_RETURNED = 200
MAX_TOP_VALUES = 5


@dataclass
class Dataset:
    name: str
    columns: list[str]
    rows: list[dict[str, Any]]
    raw_csv: str = ""


_DATASETS: dict[str, Dataset] = {}


class FilterSpec(BaseModel):
    column: str = Field(description="Nom exact de la colonne à filtrer.")
    operator: Literal[
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "starts_with",
        "ends_with",
        "in",
        "is_null",
        "not_null",
    ] = Field(description="Opérateur de comparaison à appliquer.")
    value: Any = Field(
        default=None,
        description=(
            "Valeur de comparaison. Pour 'in', une liste de valeurs. "
            "Ignoré pour 'is_null' et 'not_null'."
        ),
    )


class AggregationSpec(BaseModel):
    column: str | None = Field(
        default=None,
        description=(
            "Colonne à agréger. Laisser None uniquement pour func='count' "
            "(comptage de lignes)."
        ),
    )
    func: Literal["count", "sum", "mean", "median", "min", "max", "nunique"] = Field(
        description="Fonction d'agrégation à appliquer."
    )


def _get_dataset(session_id: str) -> Dataset:
    dataset = _DATASETS.get(session_id)
    if dataset is None:
        raise agents.ToolError(
            "Aucun jeu de données chargé pour cette conversation. "
            "Récupère d'abord le contenu du fichier CSV avec les outils du "
            "connecteur Google Drive, puis appelle load_csv avec ce contenu."
        )
    return dataset


def _convert(value: str) -> Any:
    if value is None or value == "":
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped.replace(",", "."))
    except ValueError:
        pass
    return stripped


def _detect_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_columns(dataset: Dataset, names: list[str]) -> None:
    unknown = [name for name in names if name not in dataset.columns]
    if unknown:
        raise agents.ToolError(
            f"Colonne(s) inconnue(s): {unknown}. Colonnes disponibles: "
            f"{dataset.columns}"
        )


def _row_matches(row: dict[str, Any], spec: FilterSpec) -> bool:
    actual = row.get(spec.column)
    op = spec.operator
    if op == "is_null":
        return actual is None
    if op == "not_null":
        return actual is not None
    if op == "in":
        candidates = spec.value if isinstance(spec.value, list) else [spec.value]
        return any(_values_equal(actual, candidate) for candidate in candidates)
    if op == "contains":
        return str(spec.value).lower() in str(actual).lower()
    if op == "starts_with":
        return str(actual).lower().startswith(str(spec.value).lower())
    if op == "ends_with":
        return str(actual).lower().endswith(str(spec.value).lower())
    return _compare(actual, op, spec.value)


def _values_equal(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if _is_number(actual) and _is_number(expected):
        return float(actual) == float(expected)
    if _is_number(actual) != _is_number(expected):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if actual is None or expected is None:
        return False
    if _is_number(actual) and _is_number(expected):
        left, right = float(actual), float(expected)
    elif _is_number(actual) != _is_number(expected):
        return False
    else:
        left, right = str(actual), str(expected)
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    raise agents.ToolError(f"Opérateur non supporté: {op}")


def _as_filters(filters: list[FilterSpec | dict] | None) -> list[FilterSpec]:
    return [f if isinstance(f, FilterSpec) else FilterSpec(**f) for f in filters or []]


def _apply_filters(
    dataset: Dataset, filters: list[FilterSpec | dict] | None
) -> list[dict[str, Any]]:
    specs = _as_filters(filters)
    if not specs:
        return dataset.rows
    _require_columns(dataset, [f.column for f in specs])
    return [row for row in dataset.rows if all(_row_matches(row, f) for f in specs)]


def _aggregate(values: list[Any], func: str) -> Any:
    if func == "count":
        return len(values)
    numbers = [float(v) for v in values if _is_number(v)]
    if func == "nunique":
        return len({str(v) for v in values if v is not None})
    if not numbers:
        return None
    if func == "sum":
        return sum(numbers)
    if func == "mean":
        return statistics.fmean(numbers)
    if func == "median":
        return statistics.median(numbers)
    if func == "min":
        return min(numbers)
    if func == "max":
        return max(numbers)
    raise agents.ToolError(f"Fonction d'agrégation non supportée: {func}")


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=False))
async def load_csv(
    ctx: agents.ToolCallbackContext,
    /,
    csv_content: Annotated[str, Field(description="Contenu du fichier CSV.")],
    content_encoding: Annotated[
        Literal["text", "base64"],
        Field(
            description=(
                "Encoding of csv_content. Use 'base64' when it comes from the "
                "Drive connector tool download_file_content, which returns "
                "base64-encoded content."
            )
        ),
    ] = "text",
    file_name: Annotated[str, Field(description="Nom du fichier source.")] = "dataset.csv",
) -> dict:
    """Charge un contenu CSV comme jeu de données actif de cette conversation.

    Récupère d'abord le contenu du fichier avec les outils du connecteur
    Google Drive, puis passe ce contenu ici. Remplace le jeu de données
    précédent de la conversation.
    """
    if not csv_content.strip():
        raise agents.ToolError("Le contenu CSV fourni est vide.")
    if content_encoding == "base64":
        try:
            csv_content = base64.b64decode(csv_content).decode("utf-8-sig")
        except Exception as error:
            raise agents.ToolError(
                f"Décodage base64 impossible: {error}. Vérifie que csv_content "
                "vient bien de download_file_content."
            ) from error
    try:
        dialect = _detect_dialect(csv_content[:4096])
        reader = csv.DictReader(io.StringIO(csv_content), dialect=dialect)
        columns = [
            (name or f"colonne_{i + 1}") for i, name in enumerate(reader.fieldnames or [])
        ]
        if not columns:
            raise agents.ToolError(
                "Impossible de lire l'en-tête du CSV : aucune colonne détectée."
            )
        rows = []
        for row in reader:
            rows.append(
                {
                    name: _convert(row.get(name))
                    for name in columns
                }
            )
    except agents.ToolError:
        raise
    except Exception as error:
        raise agents.ToolError(f"Analyse du CSV impossible: {error}") from error
    dataset = Dataset(name=file_name, columns=columns, rows=rows, raw_csv=csv_content)
    _DATASETS[ctx.session.id] = dataset
    return {
        "file": file_name,
        "rows": len(rows),
        "columns": columns,
        "preview": rows[:MAX_PREVIEW_ROWS],
    }


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def describe_columns(ctx: agents.ToolCallbackContext, /) -> dict:
    """Décrive les colonnes du jeu de données chargé.

    Pour chaque colonne : type détecté, valeurs présentes/manquantes, nombre de
    valeurs uniques, et statistiques (min, max, moyenne, somme) pour les
    colonnes numériques ou valeurs les plus fréquentes pour les autres.
    """
    dataset = _get_dataset(ctx.session.id)
    descriptions = []
    for column in dataset.columns:
        values = [row[column] for row in dataset.rows]
        present = [v for v in values if v is not None]
        description: dict[str, Any] = {
            "column": column,
            "type": "number" if present and all(_is_number(v) for v in present) else "string",
            "non_null": len(present),
            "null": len(values) - len(present),
            "unique": len({str(v) for v in present}),
        }
        if description["type"] == "number":
            numbers = [float(v) for v in present]
            description["stats"] = {
                "min": min(numbers),
                "max": max(numbers),
                "mean": statistics.fmean(numbers),
                "sum": sum(numbers),
            }
        elif present:
            top = Counter(str(v) for v in present).most_common(MAX_TOP_VALUES)
            description["most_frequent"] = [{"value": v, "count": c} for v, c in top]
        descriptions.append(description)
    return {"file": dataset.name, "rows": len(dataset.rows), "columns": descriptions}


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    filters: Annotated[
        list[FilterSpec], Field(description="Filtres à appliquer (ET logique entre eux).")
    ],
    limit: Annotated[
        int, Field(description="Nombre maximum de lignes renvoyées.", ge=1, le=MAX_FILTER_RETURNED_ROWS)
    ] = 20,
) -> dict:
    """Filtre les lignes du jeu de données chargé et renvoie les correspondances.

    Plusieurs filtres se combinent en ET logique. Renvoie le nombre total de
    lignes correspondantes et jusqu'à `limit` lignes d'exemple.
    """
    dataset = _get_dataset(ctx.session.id)
    matched = _apply_filters(dataset, filters)
    return {
        "file": dataset.name,
        "total_rows": len(dataset.rows),
        "matched": len(matched),
        "returned": min(len(matched), limit),
        "rows": matched[:limit],
    }


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def group_aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[
        list[str], Field(description="Colonnes de regroupement (une ou plusieurs).")
    ],
    aggregations: Annotated[
        list[AggregationSpec],
        Field(description="Agrégats à calculer par groupe, ex. somme ou moyenne d'une colonne."),
    ],
    filters: Annotated[
        list[FilterSpec] | None,
        Field(description="Filtres optionnels appliqués avant le regroupement (ET logique)."),
    ] = None,
    sort_by: Annotated[
        str | None,
        Field(description="Nom de l'agrégat ('colonne_func' ou 'count') utilisé pour trier les groupes."),
    ] = None,
    sort_desc: Annotated[bool, Field(description="Tri décroissant si vrai.")] = True,
    limit: Annotated[
        int, Field(description="Nombre maximum de groupes renvoyés.", ge=1, le=MAX_GROUP_RETURNED)
    ] = 50,
) -> dict:
    """Calcule des agrégats par groupe sur le jeu de données chargé.

    Regroupe les lignes filtrées par les colonnes `group_by` et calcule chaque
    agrégat demandé (ex. {'column': 'ventes', 'func': 'sum'}) pour chaque
    groupe. Les agrégats sont nommés 'colonne_func', ou 'count' pour le
    comptage.
    """
    dataset = _get_dataset(ctx.session.id)
    _require_columns(dataset, group_by)
    aggregations = [
        a if isinstance(a, AggregationSpec) else AggregationSpec(**a) for a in aggregations
    ]
    agg_columns = [a.column for a in aggregations if a.column is not None]
    _require_columns(dataset, agg_columns)

    rows = _apply_filters(dataset, filters)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_by)
        groups.setdefault(key, []).append(row)

    results = []
    for key, group_rows in groups.items():
        entry: dict[str, Any] = {
            column: value for column, value in zip(group_by, key)
        }
        for spec in aggregations:
            label = "count" if spec.column is None else f"{spec.column}_{spec.func}"
            values = [row[spec.column] for row in group_rows] if spec.column else group_rows
            entry[label] = _aggregate(values, spec.func)
        results.append(entry)

    if sort_by:
        if sort_by not in (results[0] if results else {}):
            raise agents.ToolError(
                f"Agrégat inconnu pour le tri: {sort_by}. Agrégats disponibles: "
                f"{list(results[0].keys()) if results else []}"
            )
        results.sort(key=lambda entry: (entry[sort_by] is None, entry[sort_by]), reverse=sort_desc)

    return {
        "file": dataset.name,
        "rows_aggregated": len(rows),
        "groups": len(results),
        "returned": min(len(results), limit),
        "results": results[:limit],
    }


agent = agents.Agent(
    model="mistral-large-4",
    name="data-analyst",
    instructions=(
        "Tu es un analyste de données conversationnel qui répond en français.\n"
        "\n"
        "Flux de travail :\n"
        "1. L'utilisateur te donne un lien vers un fichier CSV de son Google "
        "Drive. Extrais l'identifiant du fichier du lien, puis utilise les "
        "outils du connecteur Google Drive : get_file_metadata pour vérifier le "
        "fichier, et download_file_content pour récupérer son contenu (avec "
        "export_mime_type='text/csv' si c'est un Google Sheet). Le contenu "
        "renvoyé par download_file_content est encodé en base64. Si le "
        "connecteur demande une connexion Google Drive, demande à l'utilisateur "
        "de la valider.\n"
        "2. Appelle load_csv avec ce contenu et content_encoding='base64'. Le jeu "
        "de données reste "
        "chargé pour toute la conversation : ne redemande jamais le lien.\n"
        "3. Pour chaque question, utilise describe_columns pour découvrir les "
        "colonnes si nécessaire, puis filter_rows et group_aggregate pour "
        "calculer les réponses.\n"
        "\n"
        "Règles :\n"
        "- Ne réponds JAMAIS de mémoire : chaque chiffre cité doit venir du "
        "résultat d'un outil sur les données chargées.\n"
        "- Si un outil renvoie une erreur (colonne inconnue, jeu de données non "
        "chargé), corrige ta requête ou recharge le fichier, puis réessaie.\n"
        "- Présente les résultats avec les chiffres exacts renvoyés par les "
        "outils, en tableau quand c'est pertinent, et précise le nombre de "
        "lignes concernées.\n"
        "- N'exécute aucun code et n'invente aucun outil : tu ne disposes que "
        "des outils listés et de ceux du connecteur Google Drive.\n"
        "- Pour une question de suivi, réutilise le jeu de données et le "
        "contexte déjà établis dans la conversation."
    ),
    tools=[load_csv, describe_columns, filter_rows, group_aggregate],
    connectors=[agents.connector("google_drive_mcp")],
)


async def main() -> None:
    result = await agent.run(
        "Voici mon fichier : https://drive.google.com/file/d/EXEMPLE/view. "
        "Décris ses colonnes."
    )
    print(result.text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

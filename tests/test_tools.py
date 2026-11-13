# Tests des outils de l'agent analyste de données (sans appel de modèle).

import base64
import csv
import io

import pytest
from mistralai import agents

from main import (
    Dataset,
    _DATASETS,
    describe_columns,
    filter_rows,
    group_aggregate,
    load_csv,
)

CSV_CONTENT = (
    "produit,region,ventes,quantite,date\n"
    "Alpha,Nord,100,3,2024-01-15\n"
    "Beta,Sud,250,5,2024-01-16\n"
    "Alpha,Sud,120,4,2024-01-20\n"
    "Gamma,Nord,,2,2024-02-01\n"
    "Beta,Nord,300,6,2024-02-10\n"
)


@pytest.fixture(autouse=True)
def clean_store():
    _DATASETS.clear()
    yield
    _DATASETS.clear()


async def _load(**kwargs) -> dict:
    return await load_csv(csv_content=CSV_CONTENT, file_name="ventes.csv", **kwargs)


async def test_load_csv_parses_types_and_stores_per_session():
    result = await _load()
    assert result["file"] == "ventes.csv"
    assert result["rows"] == 5
    assert result["columns"] == ["produit", "region", "ventes", "quantite", "date"]
    assert result["preview"][0] == {
        "produit": "Alpha",
        "region": "Nord",
        "ventes": 100,
        "quantite": 3,
        "date": "2024-01-15",
    }
    dataset = _DATASETS["local"]
    assert dataset.rows[3]["ventes"] is None


async def test_load_csv_rejects_empty_content():
    with pytest.raises(agents.ToolError, match="vide"):
        await load_csv(csv_content="   ")


async def test_load_csv_accepts_base64_content():
    encoded = base64.b64encode(CSV_CONTENT.encode()).decode()
    result = await load_csv(csv_content=encoded, content_encoding="base64", file_name="b64.csv")
    assert result["rows"] == 5
    assert result["columns"][0] == "produit"


async def test_load_csv_rejects_invalid_base64():
    with pytest.raises(agents.ToolError, match="base64"):
        await load_csv(csv_content="!!!pas-base64!!!", content_encoding="base64")


async def test_describe_columns_stats():
    await _load()
    result = await describe_columns()
    assert result["rows"] == 5
    by_name = {c["column"]: c for c in result["columns"]}
    ventes = by_name["ventes"]
    assert ventes["type"] == "number"
    assert ventes["null"] == 1
    assert ventes["stats"]["min"] == 100
    assert ventes["stats"]["max"] == 300
    assert ventes["stats"]["sum"] == 770
    produit = by_name["produit"]
    assert produit["type"] == "string"
    top = {item["value"]: item["count"] for item in produit["most_frequent"]}
    assert top == {"Alpha": 2, "Beta": 2, "Gamma": 1}


async def test_describe_columns_requires_loaded_dataset():
    with pytest.raises(agents.ToolError, match="Aucun jeu de données"):
        await describe_columns()


async def test_filter_rows_numeric_and_contains():
    await _load()
    result = await filter_rows(filters=[{"column": "region", "operator": "eq", "value": "Nord"}])
    assert result["matched"] == 3

    result = await filter_rows(
        filters=[{"column": "produit", "operator": "contains", "value": "alp"}]
    )
    assert result["matched"] == 2
    assert all(row["produit"] == "Alpha" for row in result["rows"])

    result = await filter_rows(
        filters=[{"column": "ventes", "operator": "gt", "value": 150}]
    )
    assert result["matched"] == 2


async def test_filter_rows_combined_and_null():
    await _load()
    result = await filter_rows(
        filters=[
            {"column": "region", "operator": "eq", "value": "Nord"},
            {"column": "ventes", "operator": "not_null", "value": None},
        ]
    )
    assert result["matched"] == 2

    result = await filter_rows(filters=[{"column": "ventes", "operator": "is_null"}])
    assert result["matched"] == 1
    assert result["rows"][0]["produit"] == "Gamma"


async def test_filter_rows_unknown_column():
    await _load()
    with pytest.raises(agents.ToolError, match="inconnue"):
        await filter_rows(filters=[{"column": "prix", "operator": "eq", "value": 1}])


async def test_group_aggregate_sum_and_mean():
    await _load()
    result = await group_aggregate(
        group_by=["produit"],
        aggregations=[{"column": "ventes", "func": "sum"}, {"column": None, "func": "count"}],
    )
    by_product = {r["produit"]: r for r in result["results"]}
    assert by_product["Alpha"]["ventes_sum"] == 220
    assert by_product["Alpha"]["count"] == 2
    assert by_product["Gamma"]["ventes_sum"] is None
    assert result["rows_aggregated"] == 5

    result = await group_aggregate(
        group_by=["region"],
        aggregations=[{"column": "ventes", "func": "mean"}],
    )
    by_region = {r["region"]: r for r in result["results"]}
    assert by_region["Sud"]["ventes_mean"] == pytest.approx(185.0)
    assert by_region["Nord"]["ventes_mean"] == pytest.approx(200.0)


async def test_group_aggregate_with_filters_and_sort():
    await _load()
    result = await group_aggregate(
        group_by=["produit"],
        aggregations=[{"column": "quantite", "func": "sum"}],
        filters=[{"column": "region", "operator": "eq", "value": "Nord"}],
        sort_by="quantite_sum",
    )
    assert [r["produit"] for r in result["results"]] == ["Beta", "Alpha", "Gamma"]
    assert result["rows_aggregated"] == 3


async def test_group_aggregate_unknown_column():
    await _load()
    with pytest.raises(agents.ToolError, match="inconnue"):
        await group_aggregate(group_by=["ville"], aggregations=[{"column": None, "func": "count"}])

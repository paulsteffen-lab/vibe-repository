# Test de la boucle de l'agent avec un modèle scripté : aucune clé API requise.

import pytest
from mistralai import agents

from main import _DATASETS, agent

CSV_CONTENT = (
    "produit,region,ventes\n"
    "Alpha,Nord,100\n"
    "Beta,Sud,250\n"
)


@pytest.fixture(autouse=True)
def clean_store():
    _DATASETS.clear()
    yield
    _DATASETS.clear()


async def test_agent_turn_calls_load_csv_then_replies(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy-key-for-scripted-test")
    scripted = agent.override(
        model=agents.testing.ScriptedModel(
            [
                agents.testing.call("load_csv", csv_content=CSV_CONTENT, file_name="ventes.csv"),
                agents.testing.reply("Fichier chargé : 2 lignes."),
            ]
        )
    )
    result = await scripted.run("Analyse ce fichier CSV")
    assert isinstance(result, agents.Completed)
    assert "2 lignes" in result.text
    assert any(call.name == "client.load_csv" for call in result.tool_calls)
    assert result.session_id in _DATASETS
    assert _DATASETS[result.session_id].rows[1]["ventes"] == 250

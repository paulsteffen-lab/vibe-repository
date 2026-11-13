# data-analyst — agent analyste de données conversationnel

Agent construit avec le **SDK Agent de Mistral** (`mistralai-agents` 1.1.0), déployé en
**déploiement géré Mistral Cloud** (`agents deploy`).

Vous lui donnez un lien vers un fichier CSV de votre Google Drive, puis vous lui posez vos
questions en langage naturel. Il répond avec des chiffres réels calculés sur vos données.

## Fonctionnement

- **Google Drive** : le connecteur Studio `google_drive_mcp` est attaché à l'agent
  (`connectors=[agents.connector("google_drive_mcp")]`). Il agit avec **votre** connexion
  Drive (`run_as="auto"`), pour l'utilisateur qui ouvre la session. L'agent s'en sert pour
  retrouver le fichier à partir du lien et lire son contenu.
- **Outils fixes** (`model_access="direct"`, appelés directement par le modèle, sans écriture
  de code) :
  | Outil | Rôle |
  | --- | --- |
  | `load_csv` | charge le contenu CSV comme jeu de données de la conversation |
  | `describe_columns` | type, valeurs manquantes, uniques, stats par colonne |
  | `filter_rows` | filtre les lignes (égalité, comparaisons, contains, in, null…) |
  | `group_aggregate` | agrégats par groupe (count, sum, mean, median, min, max, nunique) |
- **Aucune exécution de code arbitraire** : pas de sandbox, pas d'outil d'exécution
  (`sandbox: none` dans `agents info`). Le modèle ne peut qu'appeler ces outils fixes et
  ceux du connecteur Drive.
- **Mémoire de conversation** : les sessions d'un agent déployé sont persistantes côté
  Mistral. Le jeu de données et l'historique restent attachés à la session : les questions
  de suivi n'exigent aucune réexplication. Si le worker a redémarré entre-temps, l'outil
  renvoie une erreur claire et l'agent recharge le fichier depuis Drive tout seul.

## Tester en local

```bash
uv sync
uv run pytest -q          # 11 tests des outils et de la boucle agent (sans clé API)
uv run agents info        # vérifie la configuration de l'agent
```

Pour un essai avec un vrai modèle : créez une clé API, mettez-la dans `.env`
(`MISTRAL_API_KEY=...`), puis `uv run --env-file .env agents chat`.

## Déployer (déploiement géré Mistral Cloud)

Prérequis :

1. **Dépôt GitHub** contenant ce projet, et le **connecteur GitHub connecté sur votre
   compte** dans [Mistral Studio](https://console.mistral.ai/build/connectors) (la
   plateforme clone le projet à travers lui).
2. **Connecteur Google Drive connecté** dans le même Studio (c'est lui qui donne à l'agent
   l'accès à vos fichiers).
3. Une **`MISTRAL_API_KEY`** dont l'utilisateur a un rôle workspace avec accès Workflows
   (Developer ou Workspace Contributor, à attribuer dans la console d'admin).

Puis, depuis le dossier du projet :

```bash
git add --all && git commit -m "Agent analyste de données" && git push -u origin HEAD
uv run --env-file .env agents deploy
```

Le déploiement est **hardened** automatiquement (l'agent agit au nom de l'utilisateur de la
session via le connecteur Drive) — ne passez pas `--no-hardened`.

## Interagir avec l'agent déployé

À la fin du déploiement, la commande affiche un lien **« Try the agent: in Vibe Code Web »**
de la forme `https://chat.mistral.ai/managed-agents/data-analyst?source=agents_api` :
ownrez-le et discutez directement avec l'agent dans Vibe. Chaque conversation est une
session persistante : donnez le lien du CSV une fois, puis enchaînez vos questions.

Autres canaux :

```bash
uv run --env-file .env agents chat data-analyst   # chat en terminal
uv run --env-file .env agents logs data-analyst --follow   # logs
```

Ou depuis du code : `agents.Agent.load_from(agent_name="data-analyst")`, puis
`agent.start_session()` et `session.run(...)` (Session API).

## Fichiers

- `main.py` — l'agent, ses instructions et ses 4 outils
- `tests/` — tests des outils et de la boucle agent (modèle scripté, sans clé API)
- `Dockerfile`, `.python-version` — image du worker de déploiement
- `pyproject.toml`, `uv.lock` — dépendances (SDK uniquement, pas de pandas : analyse en
  bibliothèque standard)

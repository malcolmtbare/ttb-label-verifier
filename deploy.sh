#!/usr/bin/env bash
# One-shot deploy to Azure Container Apps.
# Builds in the cloud from the Dockerfile, then pushes credentials from your local .env
# as Container Apps secrets/env vars. Run from the repo root after `az login`.
set -euo pipefail

# ---- edit these ----------------------------------------------------------
RG=ttb-verifier-rg
APP=ttb-verifier
ENVNAME=ttb-verifier-env
LOCATION=eastus            # use your tenant's region
# --------------------------------------------------------------------------

if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and fill in the models you use." >&2
  exit 1
fi
# Load .env into the environment
set -a; . ./.env; set +a

echo "==> Ensuring prerequisites"
az extension add --name containerapp --upgrade -o none
az provider register --namespace Microsoft.App -o none
az provider register --namespace Microsoft.OperationalInsights -o none

echo "==> Resource group"
az group create --name "$RG" --location "$LOCATION" -o none

echo "==> Build + deploy (cloud build from Dockerfile)"
az containerapp up \
  --name "$APP" --resource-group "$RG" --location "$LOCATION" \
  --environment "$ENVNAME" --source . \
  --ingress external --target-port 8000

# ---- credentials ---------------------------------------------------------
# Build secret list (keys) and env list (endpoints/models + secretref pointers)
# only for the values that are actually set in .env.
secrets=()
envs=()
add_secret() { # name value  -> secret + matching env var via secretref
  local sname="$1" ename="$2" val="${3:-}"
  if [ -n "$val" ]; then secrets+=("$sname=$val"); envs+=("$ename=secretref:$sname"); fi
}
add_env() { local name="$1" val="${2:-}"; if [ -n "$val" ]; then envs+=("$name=$val"); fi; }

add_secret azure-openai-key  AZURE_OPENAI_API_KEY    "${AZURE_OPENAI_API_KEY:-}"
add_secret azure-vision-key  AZURE_VISION_KEY        "${AZURE_VISION_KEY:-}"
add_secret foundry-key       FOUNDRY_CLAUDE_API_KEY  "${FOUNDRY_CLAUDE_API_KEY:-}"
add_secret gemini-key        GEMINI_API_KEY          "${GEMINI_API_KEY:-}"

add_env AZURE_OPENAI_ENDPOINT    "${AZURE_OPENAI_ENDPOINT:-}"
add_env AZURE_OPENAI_DEPLOYMENT  "${AZURE_OPENAI_DEPLOYMENT:-}"
add_env AZURE_VISION_ENDPOINT    "${AZURE_VISION_ENDPOINT:-}"
add_env FOUNDRY_CLAUDE_ENDPOINT  "${FOUNDRY_CLAUDE_ENDPOINT:-}"
add_env FOUNDRY_CLAUDE_MODEL     "${FOUNDRY_CLAUDE_MODEL:-}"
add_env GEMINI_MODEL             "${GEMINI_MODEL:-}"

if [ ${#secrets[@]} -gt 0 ]; then
  echo "==> Setting secrets"
  az containerapp secret set -n "$APP" -g "$RG" --secrets "${secrets[@]}" -o none
fi
if [ ${#envs[@]} -gt 0 ]; then
  echo "==> Setting environment variables"
  az containerapp update -n "$APP" -g "$RG" --set-env-vars "${envs[@]}" -o none
fi

URL=$(az containerapp show -n "$APP" -g "$RG" \
        --query properties.configuration.ingress.fqdn -o tsv)
echo
echo "Deployed: https://$URL/"
echo "Batch:    https://$URL/batch"

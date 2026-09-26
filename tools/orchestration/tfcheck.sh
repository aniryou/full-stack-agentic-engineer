#!/usr/bin/env bash
# usage: tfcheck.sh <terraform-dir>   — copies to scratch, runs fmt -check, init (local mirror), validate
set -euo pipefail
SP=${ORCH_SCRATCH:-/tmp/claude-0/orch}   # local mirror + terraform binary live here (see README)
src=$(cd "$1" && pwd)
work=$(mktemp -d -p "$SP")
cp -r "$src"/. "$work"/
rm -rf "$work/.terraform" "$work/.terraform.lock.hcl"
cd "$work"
export TF_CLI_CONFIG_FILE=${TF_CLI_CONFIG_FILE:-$SP/tf/terraformrc}
${TERRAFORM_BIN:-terraform} fmt -check -recursive -diff || { echo "FMT FAILED (run: terraform fmt -recursive $src)"; exit 1; }
${TERRAFORM_BIN:-terraform} init -backend=false -input=false -no-color >/dev/null
${TERRAFORM_BIN:-terraform} validate -no-color
rm -rf "$work"

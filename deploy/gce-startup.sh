#!/bin/bash
set -euxo pipefail
exec > >(tee -a /var/log/sumora-startup.log) 2>&1

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io nginx python3-venv python3-pip curl jq
systemctl enable --now docker
systemctl enable --now nginx

install -d -m 0755 /opt/sumora /opt/sumora/data /opt/sumora/run /var/log/sumora
systemctl is-active docker
systemctl is-active nginx

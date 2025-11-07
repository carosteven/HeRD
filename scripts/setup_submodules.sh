#!/usr/bin/env bash
set -e
git submodule update --init --recursive

pip install -e submodules/BenchNPIN
pip install -e submodules/spfa
pip install -e submodules/diffusionPolicy
#!/bin/bash

tritonserver --model-repository=/models &

fast-deploy $@
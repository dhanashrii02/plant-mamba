@echo off
title Plant Mamba Model Training
echo Starting Vision Mamba Model Training...
cd /d "%~dp0core_engine"
"C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe" train.py --data_dir ./plantvillage_data/data --epochs 5 --subset 120 --batch_size 32 --out checkpoint.pt
pause

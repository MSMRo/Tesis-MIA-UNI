#!/usr/bin/env python3
"""
Script de verificación rápida de dependencias y configuración
para el notebook VAE1.ipynb mejorado
"""

import sys
import subprocess

print("=" * 80)
print("🔍 VERIFICACIÓN DE DEPENDENCIAS - VAE CONDICIONAL MEJORADO")
print("=" * 80)

# Lista de paquetes necesarios
required_packages = {
    'torch': 'PyTorch (GPU/CPU)',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'scipy': 'SciPy',
    'tqdm': 'tqdm',
}

optional_packages = {
    'fastdtw': 'FastDTW (para DTW distance)',
}

print("\n✓ Verificando paquetes requeridos:\n")

missing_required = []
for package, display_name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✅ {display_name:<40} - Instalado")
    except ImportError:
        print(f"  ❌ {display_name:<40} - FALTA")
        missing_required.append(package)

print("\n✓ Verificando paquetes opcionales:\n")

missing_optional = []
for package, display_name in optional_packages.items():
    try:
        __import__(package)
        print(f"  ✅ {display_name:<40} - Instalado")
    except ImportError:
        print(f"  ⚠️  {display_name:<40} - FALTA (será instalado automáticamente)")
        missing_optional.append(package)

# Instalar paquetes opcionales faltantes
if missing_optional:
    print(f"\n⬇️  Instalando paquetes opcionales faltantes...")
    for package in missing_optional:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"  ✅ {package} instalado correctamente")
        except Exception as e:
            print(f"  ⚠️  Error instalando {package}: {e}")

# Verificar GPU
print("\n✓ Verificación de GPU/CUDA:\n")

try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ CUDA disponible")
        print(f"     - Device: {torch.cuda.get_device_name(0)}")
        print(f"     - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"  ⚠️  CUDA NO disponible - Se usará CPU")
except Exception as e:
    print(f"  ⚠️  Error verificando CUDA: {e}")

# Resumen final
print("\n" + "=" * 80)
if not missing_required:
    print("✅ CONFIGURACIÓN LISTA - Todos los paquetes requeridos están instalados")
    print("\n📋 Puedes ejecutar el notebook VAE1.ipynb sin problemas")
else:
    print(f"❌ FALTA INSTALAR {len(missing_required)} paquete(s) requerido(s):")
    for package in missing_required:
        print(f"   pip install {package}")
    sys.exit(1)

print("=" * 80)

"""
Setup script for Enterprise RAG Knowledge Base.
Run this script to initialize the system.
"""
import os
import sys
import subprocess
from pathlib import Path
from loguru import logger


def check_python_version():
    """Check if Python version is compatible."""
    logger.info("Checking Python version...")
    
    if sys.version_info < (3, 9):
        logger.error("Python 3.9+ is required")
        return False
    
    logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_docker():
    """Check if Docker is installed and running."""
    logger.info("Checking Docker...")
    
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✓ {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("✗ Docker not found or not running")
        logger.warning("Docker is required for Qdrant. Install from: https://www.docker.com/")
        return False


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    logger.info("Setting up environment file...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        logger.info("✓ .env file already exists")
        return True
    
    if not env_example.exists():
        logger.error("✗ .env.example not found")
        return False
    
    # Copy template
    env_file.write_text(env_example.read_text())
    logger.info("✓ Created .env file from template")
    logger.warning("⚠ Please edit .env and add your OPENAI_API_KEY")
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        logger.info("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install dependencies: {e}")
        return False


def start_qdrant():
    """Start Qdrant using Docker."""
    logger.info("Starting Qdrant...")
    
    try:
        # Check if Qdrant is already running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "qdrant" in result.stdout:
            logger.info("✓ Qdrant is already running")
            return True
        
        # Start Qdrant
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", "qdrant",
                "-p", "6333:6333",
                "-p", "6334:6334",
                "-v", f"{Path.cwd()}/qdrant_storage:/qdrant/storage:z",
                "qdrant/qdrant"
            ],
            check=True
        )
        
        logger.info("✓ Qdrant started successfully")
        logger.info("  - REST API: http://localhost:6333")
        logger.info("  - gRPC API: http://localhost:6334")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to start Qdrant: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    logger.info("Creating directories...")
    
    directories = ["logs", "data", "uploads"]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    logger.info("✓ Directories created")
    return True


def verify_openai_key():
    """Verify OpenAI API key is set."""
    logger.info("Checking OpenAI API key...")
    
    # Try to load from .env
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        logger.warning("⚠ OpenAI API key not set in .env file")
        logger.warning("  Please edit .env and add your OPENAI_API_KEY")
        return False
    
    logger.info("✓ OpenAI API key found")
    return True


def main():
    """Run setup process."""
    logger.info("=" * 60)
    logger.info("Enterprise RAG Knowledge Base - Setup")
    logger.info("=" * 60)
    
    steps = [
        ("Python Version", check_python_version),
        ("Docker", check_docker),
        ("Environment File", create_env_file),
        ("Directories", create_directories),
        ("Dependencies", install_dependencies),
        ("Qdrant", start_qdrant),
        ("OpenAI API Key", verify_openai_key),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        logger.info(f"\n[{step_name}]")
        try:
            success = step_func()
            results.append((step_name, success))
        except Exception as e:
            logger.error(f"Error in {step_name}: {e}")
            results.append((step_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Setup Summary")
    logger.info("=" * 60)
    
    for step_name, success in results:
        status = "✓" if success else "✗"
        logger.info(f"{status} {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        logger.info("\n" + "=" * 60)
        logger.info("✓ Setup completed successfully!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Edit .env and add your OPENAI_API_KEY")
        logger.info("2. Run: python main.py")
        logger.info("3. Visit: http://localhost:8000/docs")
        logger.info("4. Try examples: python examples.py")
    else:
        logger.warning("\n" + "=" * 60)
        logger.warning("⚠ Setup completed with warnings")
        logger.warning("=" * 60)
        logger.warning("Please resolve the issues above before running the application")


if __name__ == "__main__":
    main()

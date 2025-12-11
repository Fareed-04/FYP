"""
Training Module for VisionAI with MLOps Integration
Supports MLflow experiment tracking and DagsHub integration
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    logger, load_params, print_banner, ensure_dir,
    setup_dagshub, get_class_names, validate_dataset
)


def setup_experiment_tracking(
    dagshub_owner: str = None,
    dagshub_repo: str = None,
    mlflow_uri: str = None,
    experiment_name: str = None
) -> bool:
    """
    Setup experiment tracking with DagsHub and MLflow.
    
    Args:
        dagshub_owner: DagsHub username
        dagshub_repo: DagsHub repository name  
        mlflow_uri: MLflow tracking URI (optional, uses DagsHub if not provided)
        experiment_name: Name for the experiment
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        import mlflow
        
        # Initialize DagsHub if credentials provided
        if dagshub_owner and dagshub_repo:
            try:
                import dagshub
                dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
                logger.info(f"✓ DagsHub initialized: {dagshub_owner}/{dagshub_repo}")
            except Exception as e:
                logger.warning(f"DagsHub init failed: {e}")
        
        # Set MLflow tracking URI
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        elif dagshub_owner and dagshub_repo:
            uri = f"https://dagshub.com/{dagshub_owner}/{dagshub_repo}.mlflow"
            mlflow.set_tracking_uri(uri)
            logger.info(f"✓ MLflow URI: {uri}")
        
        # Set experiment name
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            logger.info(f"✓ Experiment: {experiment_name}")
        
        return True
        
    except ImportError as e:
        logger.warning(f"MLflow/DagsHub not available: {e}")
        return False


def train_yolo(
    data_yaml: str,
    model: str = "yolo11s.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    learning_rate: float = 0.01,
    patience: int = 10,
    device: str = "auto",
    project: str = "VisionAI_Runs",
    name: str = None,
    resume: bool = False,
    pretrained: bool = True,
    save_artifacts: bool = True,
    dagshub_owner: str = None,
    dagshub_repo: str = None
) -> Dict[str, Any]:
    """
    Train YOLO model with full MLOps integration.
    
    Args:
        data_yaml: Path to data.yaml file
        model: Base model to use (yolo11n.pt, yolo11s.pt, etc.)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        learning_rate: Initial learning rate
        patience: Early stopping patience
        device: Training device (auto, cpu, 0, 1, etc.)
        project: Project name for saving runs
        name: Run name (auto-generated if None)
        resume: Resume from last checkpoint
        pretrained: Use pretrained weights
        save_artifacts: Save model artifacts to MLflow
        dagshub_owner: DagsHub username for tracking
        dagshub_repo: DagsHub repository name
        
    Returns:
        Dictionary with training results and paths
    """
    from ultralytics import YOLO
    
    # Validate dataset
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")
    
    # Generate run name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{timestamp}"
    
    print_banner(f"YOLO TRAINING: {name}")
    print(f"  Model: {model}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Device: {device}")
    
    # Setup experiment tracking
    mlflow_enabled = setup_experiment_tracking(
        dagshub_owner=dagshub_owner,
        dagshub_repo=dagshub_repo,
        experiment_name="YOLOv11-Household-Objects"
    )
    
    # Load model
    yolo_model = YOLO(model)
    
    # Train the model
    # Ultralytics automatically detects MLflow and logs metrics
    results = yolo_model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        patience=patience,
        device=device if device != "auto" else None,
        project=project,
        name=name,
        resume=resume,
        pretrained=pretrained,
        save=True,
        plots=True,
        val=True,
        verbose=True
    )
    
    # Get paths to saved models
    run_dir = Path(project) / name
    best_model = run_dir / "weights" / "best.pt"
    last_model = run_dir / "weights" / "last.pt"
    
    # Log artifacts to MLflow manually if needed
    if mlflow_enabled and save_artifacts:
        try:
            import mlflow
            with mlflow.start_run(run_name=name, nested=True):
                # Log hyperparameters
                mlflow.log_params({
                    "model": model,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "img_size": img_size,
                    "learning_rate": learning_rate,
                    "patience": patience
                })
                
                # Log best model
                if best_model.exists():
                    mlflow.log_artifact(str(best_model), "models")
                    logger.info(f"✓ Logged best.pt to MLflow")
                
                # Log training curves
                plots_dir = run_dir
                for plot_file in plots_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot_file), "plots")
                
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")
    
    # Prepare results dictionary
    output = {
        "run_dir": str(run_dir),
        "best_model": str(best_model) if best_model.exists() else None,
        "last_model": str(last_model) if last_model.exists() else None,
        "metrics": {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        }
    }
    
    print_banner("TRAINING COMPLETE")
    print(f"  Run Directory: {output['run_dir']}")
    print(f"  Best Model: {output['best_model']}")
    print(f"  mAP@50: {output['metrics']['mAP50']:.4f}")
    print(f"  mAP@50-95: {output['metrics']['mAP50-95']:.4f}")
    print(f"  Precision: {output['metrics']['precision']:.4f}")
    print(f"  Recall: {output['metrics']['recall']:.4f}")
    
    return output


def train_from_params(params_path: str = "params.yaml", **overrides) -> Dict[str, Any]:
    """
    Train using parameters from params.yaml file.
    
    Args:
        params_path: Path to parameters file
        **overrides: Override specific parameters
        
    Returns:
        Training results dictionary
    """
    params = load_params(params_path)
    
    # Extract parameters
    train_config = {
        "model": params.get("model", {}).get("architecture", "yolo11s.pt"),
        "epochs": params.get("train", {}).get("epochs", 50),
        "batch_size": params.get("train", {}).get("batch_size", 16),
        "img_size": params.get("model", {}).get("input_size", 640),
        "learning_rate": params.get("train", {}).get("learning_rate", 0.01),
        "patience": params.get("train", {}).get("patience", 10),
        "data_yaml": params.get("data", {}).get("config_path", "data.yaml"),
        "dagshub_owner": params.get("experiment", {}).get("dagshub_owner"),
        "dagshub_repo": params.get("experiment", {}).get("dagshub_repo"),
    }
    
    # Apply overrides
    train_config.update(overrides)
    
    return train_yolo(**train_config)


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO model for VisionAI with MLOps integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default parameters from params.yaml
  python train.py --data data.yaml
  
  # Train with custom epochs and batch size
  python train.py --data data.yaml --epochs 100 --batch 32
  
  # Train with DagsHub tracking
  python train.py --data data.yaml --dagshub-owner arsal6010 --dagshub-repo VisionAI
  
  # Use params.yaml configuration
  python train.py --from-params
        """
    )
    
    # Data configuration
    parser.add_argument('--data', '-d', type=str, help='Path to data.yaml')
    parser.add_argument('--from-params', action='store_true', help='Use params.yaml')
    
    # Model configuration
    parser.add_argument('--model', '-m', type=str, default='yolo11s.pt',
                        help='Base model (yolo11n.pt, yolo11s.pt, yolo11m.pt)')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Training configuration
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device (auto, cpu, 0, 1, cuda:0)')
    parser.add_argument('--project', type=str, default='VisionAI_Runs', help='Project name')
    parser.add_argument('--name', '-n', type=str, default=None, help='Run name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
    # MLOps configuration
    parser.add_argument('--dagshub-owner', type=str, help='DagsHub username')
    parser.add_argument('--dagshub-repo', type=str, help='DagsHub repo name')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow logging')
    
    args = parser.parse_args()
    
    if args.from_params:
        # Train from params.yaml
        overrides = {}
        if args.data:
            overrides['data_yaml'] = args.data
        if args.epochs != 50:
            overrides['epochs'] = args.epochs
        if args.batch != 16:
            overrides['batch_size'] = args.batch
        
        results = train_from_params(**overrides)
    else:
        # Train with CLI arguments
        if not args.data:
            parser.error("--data is required (or use --from-params)")
        
        results = train_yolo(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            learning_rate=args.lr,
            patience=args.patience,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            save_artifacts=not args.no_mlflow,
            dagshub_owner=args.dagshub_owner,
            dagshub_repo=args.dagshub_repo
        )
    
    print("\n✅ Training completed successfully!")
    print(f"Best model saved to: {results['best_model']}")


if __name__ == "__main__":
    main()


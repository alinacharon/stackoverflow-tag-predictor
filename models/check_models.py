import joblib
import pickle
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_and_resave_models():
    """Checks and resaves models for better compatibility"""
    try:
        # Define model paths
        current_dir = Path(__file__).parent
        model_path = current_dir / 'model.pkl'
        mlb_path = current_dir / 'mlb.pkl'

        logger.info(f"Checking models in directory: {current_dir}")
        logger.info(f"Path to model.pkl: {model_path}")
        logger.info(f"Path to mlb.pkl: {mlb_path}")

        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(
                f"File model.pkl not found at path: {model_path}")
        if not mlb_path.exists():
            raise FileNotFoundError(
                f"File mlb.pkl not found at path: {mlb_path}")

        # Try loading models using different methods
        logger.info("\nTrying to load model.pkl:")
        try:
            # Try loading with joblib
            model = joblib.load(model_path)
            logger.info(
                f"✓ model.pkl successfully loaded via joblib. Type: {type(model)}")
        except Exception as e:
            logger.error(f"Error loading with joblib: {e}")
            try:
                # Try loading with pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(
                    f"✓ model.pkl successfully loaded via pickle. Type: {type(model)}")
            except Exception as e:
                logger.error(f"Error loading with pickle: {e}")
                raise

        logger.info("\nTrying to load mlb.pkl:")
        try:
            # Try loading with joblib
            mlb = joblib.load(mlb_path)
            logger.info(
                f"✓ mlb.pkl successfully loaded via joblib. Type: {type(mlb)}")
        except Exception as e:
            logger.error(f"Error loading with joblib: {e}")
            try:
                # Try loading with pickle
                with open(mlb_path, 'rb') as f:
                    mlb = pickle.load(f)
                logger.info(
                    f"✓ mlb.pkl successfully loaded via pickle. Type: {type(mlb)}")
            except Exception as e:
                logger.error(f"Error loading with pickle: {e}")
                raise

        # Resave models with improved compatibility
        logger.info("\nResaving models with improved compatibility:")

        # Create backups
        model_backup = current_dir / 'model.pkl.backup'
        mlb_backup = current_dir / 'mlb.pkl.backup'

        if not model_backup.exists():
            import shutil
            shutil.copy2(model_path, model_backup)
            logger.info(f"Created backup of model.pkl")

        if not mlb_backup.exists():
            import shutil
            shutil.copy2(mlb_path, mlb_backup)
            logger.info(f"Created backup of mlb.pkl")

        # Save models using joblib with protocol 4
        joblib.dump(model, model_path, protocol=4)
        logger.info(f"✓ model.pkl resaved using joblib (protocol=4)")

        joblib.dump(mlb, mlb_path, protocol=4)
        logger.info(f"✓ mlb.pkl resaved using joblib (protocol=4)")

        logger.info("\nCheck completed successfully!")

    except Exception as e:
        logger.error(f"Error during model check: {e}")
        raise


if __name__ == "__main__":
    check_and_resave_models()

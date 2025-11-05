import argparse
import yaml
import uvicorn
from agrovision_precision.api.fastapi_server import FastAPIServer
from agrovision_precision.utils.config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description='AgroVision Precision: AI for Agriculture')
    parser.add_argument('--mode', type=str, choices=['api', 'train', 'inference', 'dashboard'], required=True, help='Operation mode')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--image', type=str, help='Input image path for inference')
    parser.add_argument('--analysis_type', type=str, choices=['disease', 'soil', 'yield', 'irrigation'], help='Analysis type for inference')
    
    args = parser.parse_args()
    
    config_loader = ConfigLoader(args.config)
    
    if args.mode == 'api':
        print("Starting AgroVision Precision API Server...")
        server = FastAPIServer(config_loader)
        server.run()
    
    elif args.mode == 'train':
        print("Starting model training...")
        from train import main as train_main
        train_main()
    
    elif args.mode == 'inference':
        print("Running inference...")
        from inference import main as inference_main
        
        if not args.image or not args.analysis_type:
            print("Error: --image and --analysis_type are required for inference mode")
            return
        
        inference_args = argparse.Namespace()
        inference_args.image = args.image
        inference_args.analysis_type = args.analysis_type
        inference_args.output = f"results_{args.analysis_type}.json"
        
        inference_main()
    
    elif args.mode == 'dashboard':
        print("Starting AgroVision Dashboard...")
        from dashboard.app import run_dashboard
        run_dashboard()

if __name__ == "__main__":
    main()
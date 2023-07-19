#!/usr/bin/env python3
import argparse
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to model file', type=str, default=None)
    args = parser.parse_args()
    
    providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(args.model_path, providers=providers)

    print("inputs")
    for idx, inp in enumerate(sess.get_inputs()):
        print(f"  {idx:02d}: name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

    print("outputs")
    for idx, oup in enumerate(sess.get_outputs()):
        print(f"  {idx:02d}: name: {oup.name}, shape: {oup.shape}, type: {oup.type}")
    

if __name__ == "__main__":
    main()


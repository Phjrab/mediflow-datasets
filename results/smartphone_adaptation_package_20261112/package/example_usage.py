from inference import load_predictor
import json,sys
predictor=load_predictor()
for path in sys.argv[1:]:
    print(json.dumps(predictor.predict_file(path),ensure_ascii=False))

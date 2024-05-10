# Retrieval Augmented Autoformalization with Refinement (Auto-correction)

## How to set up IsarMathLib in Isabelle
First, you should download [Isabelle](https://isabelle.in.tum.de/) and add its "bin" to your PATH variable. In Linuxs systems, run
```
export PATH=$Isabelle/bin:$PATH
```
where "$Isabelle" is your Isabelle directory path.

Next, copy IsarMathLib under this repository to Isabelle by running
```
cp -r IsarMathLib $Isabelle/src/ZF
```

Finally, append "ROOT" file under IsarMathLib to "ROOT" file under ZF by running
```
cd $Isabelle/src/ZF
cat IsarMathLib/ROOT >> ROOT
```

Now you should be able to build an "IsarMathLib" session in which you can use all theories from IsarMathLib.

## Results
We provide our results in Google Drive: [BM25 retrieval](https://drive.google.com/file/d/1ndmy06gRaFU2zeqrYQgxHp8erbmz3ypV/view?usp=drive_link), [Baselines](https://drive.google.com/file/d/1TpcLvNXz75yQcdggOn1oPbaq8gO_zIZ3/view?usp=drive_link), [Round 0](https://drive.google.com/file/d/1kDHCEniZYLGq9iUm2_Z5PP2laV_8Kn8u/view?usp=drive_link), [Round 1](https://drive.google.com/file/d/17E0ktPQ2F_R1T3gXA3xOHN7wPOYWRYX7/view?usp=drive_link), [Round 2](https://drive.google.com/file/d/14qT4MVL0GAGk64o5-IB4Vx-D8JE_Q7pN/view?usp=drive_link).

## Python Files Example Usage
#### IsarMathLib Extraction
```
python IsarMathLib_extraction.py
```
#### Informalization with Mistral
```
python informalization.py --model_id mistralai/Mistral-7B-Instruct-v0.2 --data_folder data/IsarMathLib/mistral_inf
```
#### Knowledge Base Generation
```
python gen_KB.py --data_folder data/IsarMathLib/mistral_inf
```
#### Retrieval Examples from Knowledge Base
```
python retrieval.py --json_file results/mistral_0_auto.json --mode 0 --org_json data/IsarMathLib/mistral_inf/train.json --kb_folder data/KB/text --retrieval_folder results/BM25_retrieval_t_0
```
#### Autoformalization
This python file has three options for mode: 0 is zero-shot setting, 1 is few-shot setting, 2 is few-shot setting with retrieved examples.
```
python autoformalization.py --model_name mistral --mode 0 --result_json results/mistral_0_auto.json --test_json data/IsarMathLib/mistral_inf/test.json
python autoformalization.py --model_name mistral --mode 1 --result_json results/mistral_1_auto.json --test_json data/IsarMathLib/mistral_inf/test.json --shot_json data/IsarMathLib/3-shot.json
python autoformalization.py --model_name mistral --mode 2 --result_json results/mistral_t_auto_0.json --test_json data/IsarMathLib/mistral_inf/test.json --retrieval_folder results/BM25_retrieval_t_0
```
#### Refinement
```
python refinement.py --model_name mistral --round 1D --result_json results/mistral_t_0_1D.json --test_json results/mistral_t_auto_0.json --shot_json data/IsarMathLib/3-shot.json --retrieval_folder results/BM25_retrieval_t_0
python refinement.py --model_name mistral --round 2 --result_json results/mistral_t_0_2.json --test_json results/mistral_t_0_1D.json --shot_json data/IsarMathLib/3-shot.json --retrieval_folder results/BM25_retrieval_t_0
```
#### Evaluation
Evaluate results by choosing from 5 metrics: BLEU, ChrF, RUBY, CodeBERTScore, Pass. The evaluation can be done with and without postprocessing.
```
python test.py --ref_json data/IsarMathLib/extraction/test.json --result_json results/mistral_t_auto_0.json --metric BLEU ChrF RUBY CodeBERTScore Pass
python test.py --ref_json data/IsarMathLib/extraction/test.json --result_json results/mistral_t_auto_0.json --metric BLEU ChrF RUBY CodeBERTScore Pass --no_post
```

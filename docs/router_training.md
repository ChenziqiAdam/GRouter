# Router Training
This is a guideline for building GRouter for multi-agent system.

## Prepare Dataset
Download the training data and eval data from huggingface, and process it into compatible format as the GDesiner Repo original needed.

## Agent
To support new logger, the output format of the llm and agent should be adjusted accordingly. You can look at [GDesigner/agents/math_solver.py](https://github.com/szjiozi/GRouter/blob/762fce5d1f68347419a232ba710835f1b11836ca/GDesigner/agents/math_solver.py#L50C5-L70C24) for examples. Generally you should only need to adjust the async_execute function of the agent.

## Train Seperate GVAE on Different Model
Create your training code according to [experiments/train_gsm8k.py](https://github.com/szjiozi/GRouter/blob/main/experiments/train_gsm8k.py), and you can run the experiment like [scripts/train_gsm8k.sh](https://github.com/szjiozi/GRouter/blob/main/scripts/train_gsm8k.sh) on the training dataset. But you have to adjust the agent_nums and other default arguments as the original experiment file shows. The training of each GVAE only takes 40 training data, which is controlled by `num_iterations` and `batch_size`. You should also set the `dataset_json` to the path you want to use as the training data.

## Build Router Training Data
Training data of the Router is built by running the trained GVAE from the last step on the next 120 data from the training dataset. This is controlled by `dataset_start_index`(which is the start of the data to train) and `num_of_data`(which is the number of the data used to train from start index). It still calls the same train_gsm8k.py, just the argument is different. A sample experiment script can be seen at [build_router_data_gsm8k.sh](https://github.com/szjiozi/GRouter/blob/main/scripts/build_router_data_gsm8k.sh). Also remember to set the `dataset_json` to the path you want to use as the training data.

## Train Router
Router is trained using this python script: [GDesigner/router/train_router.py](https://github.com/szjiozi/GRouter/blob/main/GDesigner/router/train_router.py). No need to change here.

## Test Router
The GRoutet then can be ran using [scripts/test_router_inference_gsm8k.sh](https://github.com/szjiozi/GRouter/blob/main/scripts/test_router_inference_gsm8k.sh), which will call [experiments/run_router_gsm8k.py](https://github.com/szjiozi/GRouter/blob/main/experiments/run_router_gsm8k.py) to run the experiment. Note this time the `dataset_json` should be set to the path of the test dataset.
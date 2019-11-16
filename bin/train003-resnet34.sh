model=model003-resnet34
gpu=0
fold=2
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --debug --fold ${fold} --gpu ${gpu}

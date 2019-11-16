model=model001-resnet34
gpu=1
fold=0
conf=./conf/${model}.py

python -m src.cnn.main train ${conf} --debug --fold ${fold} --gpu ${gpu}

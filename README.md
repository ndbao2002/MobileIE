### Preparation

1. Replace the dataset path in the config file.
2. If you want to train the model, set the type in config to "original" and need_slims to "false".
3. If you want to test the pretrain model, set the type in config to "re-parameterized", need_slims to "true", and load the re-parameterized pre-trained model. You can also run inference with TFLite model by executing "test_TFLite_RGB.py/test_TFLite_ISP.py".
4. You can use the TFLite model and import it into AI Benchmark (https://ai-benchmark.com/) to obtain the inference speed on mobile devices.
5. If you want to perform UIE task, replace the dataset path in config/lle.yaml with your underwater image dataset.

### Train

```bash
python main.py -task train -model_task lle/isp -device cuda
```

### Test

```bash
python main.py -task test -model_task lle/isp -device cuda
```

### Demo

```bash
python main.py -task demo -model_task lle/isp -device cuda
```

### Training instruction
```bash
python main.py -task train -model_task isp -device cuda
python main.py -task test -model_task isp -device cuda # This is for testing the ISP model, remember to change the config file to load the pre-trained ISP model.
python evaluate.py path/to/target/dir path/to/gt/dir  --resize_mode error # This is for evaluating the ISP model performance.. 
```

We doing multiple step because the PSNR/SSIM performance will different when we directly calculate the metrics on the output of the ISP model vs the saved images after post-processing. Results should be calculated after post-processing.

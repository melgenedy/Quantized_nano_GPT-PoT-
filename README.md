This code is based on NanoGPT repositoray,
https://github.com/karpathy/nanoGPT


The main readme for nanoGPT is renamed to "README_NanoGPT.md"

The main files that contain my contribution in quantizing both character level model (trained on Shakespeare dataset), and GPT-2 are following,

- sample_PoT_CL.py ==> to test quantized model of character based trained on Shakespeare dataset. Both normal quantization and power of two (PoT) are supported
- sample_FixedPoint_GPT2.py ==> to test quantized model of GPT-2 with OpenWebText dataset. Normal quantization is supported
- sample_PoT_GPT2.py ==> to test quantized model of GPT-2 with OpenWebText dataset. Power of two (PoT) is supported

Note that before running above scripts, you have to go through original repositoray "prepare" scripts to prepare data used for testing. For the charachter based model, training script is also required to generate the model, while GPT-2 is just downloaded from pretrained.

Some other scripts renamed with ending "1" to simplify debugging.

Udated repo link,

https://github.com/melgenedy/Quantized_nano_GPT-PoT-
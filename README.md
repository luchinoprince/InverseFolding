# InverseFolding

We have three notebooks for the moment. ESM is the onotebook in which I created the encoded proteins using the pretrained model. _New_Decoder_nobatch_ is a functioning notebook with does not have batchng though. There is a batching function, but simply returns a list of samples, so no effective batching really. By working I mean that it runs, and seems to give sensible results. One thing to check, not sure if the EncoderTransformerLayer treats also the single observation as a batched observation. If that is the case we have to expand the input tensor to a 3D tensor with first dimension 1 and specify that the first dimension is the batching one. In the code _New_Decoder_batch_ I am trying to implement batching trough padding, currently the code does not work but the overall architecture is there. I need to make sure that the I am properly using the padding mask to make sure that it does not affect learning, especially how to mask the parts regarding the potts model. The codes are still messy, but should be commented enough to be able to be decriptable(I hope).

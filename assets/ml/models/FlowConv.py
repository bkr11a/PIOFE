__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf
# CNN Custom Class
# TODO - FIX THIS MESS!

# Should I pass the architecture through as a dictionary instead?
class FlowConvNet(tf.keras.Model):
    def __init__(self, lb, ub, inputDim = (1,), outputDim = 1, numFilters = [], activationFunctions = [], unknownParams = [], numDownsamples = 0, scaling = False, twoFrameInput = True, flowInput = False, **kwargs):
        super().__init__(**kwargs)
        assert len(activationFunctions) == len(numFilters), "The length of the list for the activation functions should be the same as the number of layers"

        self.ub = ub
        self.lb = lb
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.flowInput = flowInput
        self.flowInputDim = (inputDim[0], inputDim[1], 2)
        self.numLayers = len(numFilters)
        self.numFilters = numFilters
        self.activationFunctions = activationFunctions
        self.numDownsamples = numDownsamples
        self.unknownParams = unknownParams
        self.twoFrameInput = twoFrameInput
        self.scaling = scaling

        if len(self.unknownParams) > 0:
            self.paramEstimations = [tf.Variable(self.unknownParams[i], trainable = True, dtype = 'float32') for i in range(len(self.unknownParams))]
            # self.sigma = unknownParams[0]
            # self.rho = unknownParams[1]
            # self.beta = unknownParams[2]

        self.scale = tf.keras.layers.Lambda(lambda x : 2.0*(x - self.lb) / (self.ub - self.lb) - 1.0)
        self.cnns = [tf.keras.layers.Conv2D(filters = numFilters[i], kernel_size = (3, 3), padding='same', activation = activationFunctions[i]) for i in range(self.numLayers)]
        self.pools = [tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'same') for i in range(numDownsamples)]
        self.upsamples = [tf.keras.layers.UpSampling2D(size = (2, 2)) for i in range(numDownsamples)]
        self.out = tf.keras.layers.Conv2D(filters = 2, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01))
        
        self.hidden = []
        for i in range(len(self.cnns)):
            if i < 0.5 * len(self.cnns):
                self.hidden.append(self.cnns[i])
                self.hidden.append(self.pools[i])
            else:
                # Repeat the block, but now to upsample
                self.hidden.append(self.upsamples[i])
                self.hidden.append(self.cnns[i])
                
        self.hidden = [tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'same'),
                       tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.UpSampling2D(size = (2, 2)),
                       tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01))]
    
    # @tf.function
    # def call(self, X):
        # Should just hand craft it here rather than have the customisability for experimentation?
        
        # Have some form of aggregation network for the combination of two images here?
    #     if self.twoFrameInput:
            # Is subtraction good? Should have a small inference/combination network here instead?
            # Take the cross correlation between the two images instead?
           
    #         Z1 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 0])
    #         Z2 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 1])
            
    #         print(X[:, 0].shape)
        
    #         Z = tf.concat([Z1, Z2], axis = -1)
            
                  
    #     if self.flowInput:
    #         I = X[0]
    #         warped = X[2]
    #         print(I.shape)
    #         print(X[1].shape)
            
            # Flow field to refine
    #         Z0 = tf.keras.layers.InputLayer(input_shape = self.flowInputDim)(X[1])
            
            # Image pair to calculate optical flow
    #         Z1 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(I[:, 0])
    #         Z2 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(I[:, 1])
            
            # Warped image from previous optical flow estimation
    #         Z3 = tf.keras.layers.InputLayer(input_shape=self.inputDim)(warped)
        
    #         Z = tf.concat([Z0, Z1, Z2, Z3], axis = -1)
            # Z = tf.concat([Z0, Z1, Z2], axis = -1)
            
            # Z4 = tf.nn.convolution(input = Z1, filters = Z2, strides=(1,1), padding='SAME', dilations=None)
            # Z = tf.concat([Z0, Z4], axis = -1)
        
     #    if self.scaling:
     #        Z = self.scale(Z)
        
     #    print("Hidden Layers!")
     #    for layer in self.hidden:
     #        print(layer)
     #        print(Z.shape)
     #        Z = layer(Z)
        
     #    print(Z.shape)

     #    out = self.out(Z)
     #    print(Z.shape)

     #    return out        

    @tf.function
    def call(self, X):
        # Should just hand craft it here rather than have the customisability for experimentation?
        
        # Have some form of aggregation network for the combination of two images here?
        if self.twoFrameInput:
            Z1 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 0])
            Z2 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 1])
        
            Z = tf.concat([Z1, Z2], axis = -1)
             
        if self.flowInput:
            I = X[0]
            warped = X[2]
            errors = X[3]
            
            # Flow field to refine
            Z0 = tf.keras.layers.InputLayer(input_shape = self.flowInputDim)(X[1])
            
            # Image pair to calculate optical flow
            Z1 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(I[:, 0])
            Z2 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(I[:, 1])
            
            # Warped image from previous optical flow estimation
            Z3 = tf.keras.layers.InputLayer(input_shape=self.inputDim)(warped)
            
            # # Calculated Error
            # Z4 = tf.keras.layers.subtract([Z2, Z3])
            Z4 = tf.keras.layers.InputLayer(input_shape=self.inputDim)(errors)
        
            # Z = tf.concat([Z0, Z1, Z2, Z3], axis = -1)
            Z = tf.concat([Z0, Z1, Z2, Z3, Z4], axis = -1)
                    
        if self.scaling:
            Z = self.scale(Z)
        
        for layer in self.hidden:
            Z = layer(Z)
        
        # Output
        out = self.out(Z)
        
        return out
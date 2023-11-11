lgraph = layerGraph();

tempLayers = [
    sequenceInputLayer([128 128 1],"Name","sequence","MinLength",50)
    sequenceFoldingLayer("Name","seqfold")];

lgraph = addLayers(lgraph,tempLayers);

tempLayers = [

    % Layer 1a
    convolution2dLayer([20 20],3,"Name","conv_1a","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_1a")
    leakyReluLayer(0.01,"Name","leakyrelu_1")

    % Layer 2a
    convolution2dLayer([7 7],3,"Name","conv_2a","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_2a")
    leakyReluLayer(0.01,"Name","leakyrelu_2")

    % Layer 3a
    convolution2dLayer([3 3],3,"Name","conv_3a","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_3a")
    leakyReluLayer(0.01,"Name","leakyrelu_3")];

lgraph = addLayers(lgraph,tempLayers);

tempLayers = [

    % Layer 1b
    convolution2dLayer([4 4],3,"Name","conv_1b","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_1b")
    reluLayer("Name","relu_1b")
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])

    % Layer 2b
    convolution2dLayer([3 3],3,"Name","conv_2b")
    batchNormalizationLayer("Name","batchnorm_2b")
    reluLayer("Name","relu_2b")

    % Layer 3b
    convolution2dLayer([3 3],3,"Name","conv_3b","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_3b")
    reluLayer("Name","relu_3b")
    maxPooling2dLayer([3 3],"Name","pool2")];

lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,'Name','concat_1')
    flattenLayer("Name","flatten")
    fullyConnectedLayer(100,"Name","fc_1")
    reluLayer("Name","relu_4")
    dropoutLayer(0.1,"Name","dropout_1")
    fullyConnectedLayer(100,"Name","fc_2")
    reluLayer("Name","relu_5")
    dropoutLayer(0.1,"Name","dropout_2")
    ];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    sequenceUnfoldingLayer("Name","sequnfold")
    lstmLayer(125,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(2,"Name","fc_3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
    %classificationLayer("Name","weighted_cross_entropy",...
    %"Classes",{'0','1'},"ClassWeights",[1,5])
    ];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,'leakyrelu_3','concat_1/in1');
lgraph = connectLayers(lgraph,'pool2','concat_1/in2');
lgraph = connectLayers(lgraph,"seqfold/out","conv_1a");
lgraph = connectLayers(lgraph,"seqfold/out","conv_1b");
lgraph = connectLayers(lgraph,...
    "seqfold/miniBatchSize","sequnfold/miniBatchSize");
lgraph = connectLayers(lgraph,"dropout_2","sequnfold/in");

plot(lgraph);
analyzeNetwork(lgraph);

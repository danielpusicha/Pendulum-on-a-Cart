clear
clc

linearizedController

mdl = 'mySimNN';
open_system(mdl)


numSims = 2;
simulations(numSims)

dataFolder = "./simulation_data/sample Time 0.01";
[U,Y] = processTrainData(dataFolder, numSims);

if numSims > 10
    hpartition = cvpartition(size(U,1), 'Holdout', 0.2);
    idxTrain = training(hpartition);
    idxTest = test(hpartition);

    UTrain = U(idxTrain);
    YTrain = Y(idxTrain);

    UTest  = U(idxTest);
    YTest  = Y(idxTest);
else
    UTrain = U;
    YTrain = Y;
    
    UTest = U;
    YTest = Y;
end
numFeatures = size(UTrain{1},1);
numResponses = size(YTrain{1},1);


numHiddenUnits = 200;

% layers = [ ...
%     sequenceInputLayer(numFeatures)
%     lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     fullyConnectedLayer(50)
%     dropoutLayer(0.5)
%     fullyConnectedLayer(numResponses)
%     regressionLayer];

layers = [ ...
    sequenceInputLayer(numFeatures)
    gruLayer(100,'Name','gru1')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 120;
miniBatchSize = 30;

options = trainingOptions('adam', ...
    'Plots','training-progress',...
    'Verbose',true,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ...
    'ValidationData',{UTest,YTest},...
    'ValidationPatience',6,...
    'InitialLearnRate',0.1, ...
    ...'LearnRateSchedule','piecewise',...
    'GradientThreshold',1);

[net, info] = trainNetwork(UTrain,YTrain,layers,options);
disp(info.TrainingLoss(end))
save('trainedNet.mat', 'net');
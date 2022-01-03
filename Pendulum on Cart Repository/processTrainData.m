function [predictors,responses] = processTrainData(dataFolder, numSims)

obs = 1;
i = 1;

while i <= numSims
    pathPredictor = dataFolder + "/U";
    pathResponse  = dataFolder + "/Y";
    
    filenamePredictors = fullfile(pathPredictor,sprintf("seed%d.mat",obs));
    filenameResponse   = fullfile(pathResponse,sprintf("seed%d.mat",obs));
    
    if exist(filenamePredictors) == 2 
        
        dataPredict = open(filenamePredictors);
        dataPredict = dataPredict.U;

        dataResponse = open(filenameResponse);
        dataResponse = dataResponse.Y;

        predictors{i} = dataPredict();
        responses{i}  = dataResponse();
        
        i = i+1;

    end
    
    obs = obs+1;
    
end

predictors = predictors';
responses = responses';
end

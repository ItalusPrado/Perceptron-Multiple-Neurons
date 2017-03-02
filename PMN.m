clc;
clear all;
close all;

%Reading file and initial values
bias = 1;
learningRate = 0.5;
seasons = 30;
data = load('iris.txt');
biasArray(1:size(data),1) = bias;

for execucoes = 1:10
    %Randomize and separate test from train
    data = data(randperm(size(data,1)),:);
    trainSize = round(size(data,1)/10*8);
    trainArray = data(1:trainSize,:);
    testArray = data(trainSize+1:size(data,1),:);
    
    %Separating result from info
    [lines,collumns] = size(data);
    typesTrain = trainArray(:,collumns-2:collumns);
    typesTest = testArray(:,collumns-2:collumns);
    
    infoTrain = [biasArray(1:trainSize,:) trainArray(:,1:collumns-3)];
    infoTest = [biasArray(trainSize+1:lines,:) testArray(:,1:collumns-3)];
    
    %Choosing weights
    [lines,collumns] = size(infoTrain);
    weights(1:collumns,1:3) = 0;
    %weights = rand(5,3);
    
    %Perceptron Train
    for i = 1:seasons
        for j = 1:lines
            resultTrain = infoTrain(j,:)*weights;
            logsig(resultTrain);
            [valueMax,indexMax] = max(resultTrain);
            for k = 1:size(resultTrain,2)
                if k == indexMax
                    resultTrain(k) = 1;
                else
                    resultTrain(1,k) = 0;
                end
            end
            
            error = typesTrain(j,:)-resultTrain;
            weights = weights+learningRate*infoTrain(j,:)'*error;
        end
    end
    
    %Perceptron Test
    acertos = 0;
    
    for i = 1:size(typesTest)
        resultTest = infoTest(i,:)*weights;
        logsig(resultTest);
        [valueMax,indexMax] = max(resultTest);
        for k = 1:size(resultTest,2)
            if k == indexMax
                resultTest(k) = 1;
            else
                resultTest(1,k) = 0;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        error = typesTest(i,:)-resultTest;
        tf = isequal(error,[0,0,0]);
        if tf
            acertos = acertos+1;
        end
    end
    totalAcertos(execucoes) = acertos;
end
bar(totalAcertos)
axis ([0 11 0 30])
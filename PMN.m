clc
clear all
close all

%Reading file and initial values
bias = 1;
learningRate = 0.5;
seasons = 10;
data = load('iris.txt');
biasArray(1:150,1) = bias;

%Randomize and separate test from train
data = data(randperm(size(data,1)),:);
trainSize = 120;%round(size(data,1)/10*2);
trainArray = data(1:trainSize,:);
testArray = data(trainSize+1:size(data,1),:);

[lines,collumns] = size(trainArray);
%Separating result from info
typesTrain = trainArray(:,collumns-2:collumns);
typesTest = testArray(:,collumns-2:collumns);

infoTrain = [biasArray(1:120,:) trainArray(:,1:collumns-3)];
infoTest = [biasArray(121:150,:) testArray(:,1:collumns-3)];

%Choosing weights
[lines,collumns] = size(infoTrain);
weights(1:collumns,1:3) = 0;
%weights = rand(5,3);

%Perceptron Train
arrayAcertos = [];
for i = 1:10
    for j = 1:120
        result = infoTrain(j,:)*weights;
        for index = 1:size(weights,2)
            if result(1,index) >= 0
                result(1,index) = 1;
            else
                result(1,index) = 0;
            end
        end
        
        error = typesTrain(j,:)-result;
        weights = weights+learningRate*infoTrain(j,:)'*error;
    end
    
%Perceptron Test

    acertos = 0;
    for j = 1:30
        result = infoTest(j,:)*weights;
        for index = 1:size(weights,2)
            if result(1,index) >= 0
                result(1,index) = 1;
            else
                result(1,index) = 0;
            end
        end
        error = typesTest(j,:)-result;
        tf = isequal(error,[0,0,0]);
        if tf
            acertos = acertos+1;
        end
    end
    arrayAcertos(end+1) = acertos;
end
bar(arrayAcertos)
axis ([0 12 0 30])

clear; %RMSE, relaxation materials, 3 descriptors (Al, In, Ga);

TrainGen = csvread('../train/relaxation/general.csv', 1);
TrainNMat = size(TrainGen,1);
TrainIndMat = TrainGen(:,1);
TrainEnergy = csvread('../train/relaxation/energy_relaxation.csv',1);

TestGen = csvread('../test/relaxation/general.csv', 1);
TestNMat = size(TestGen,1);
TestIndMat = TestGen(:,1);
TestEnergy = csvread('../test/relaxation/energy_relaxation.csv',1);

for j=1:size(TrainEnergy,1)
    idx=find(TrainIndMat==TrainEnergy(j,1));
    TrainData(j,1:3)=TrainGen(idx,4:6);
end

LSCoef=TrainData\TrainEnergy(:,3);

for j=1:size(TestEnergy,1)
    idx=find(TestIndMat==TestEnergy(j,1));
    TestData(j,1:3)=TestGen(idx,4:6);
end

TestEnergy_guess=TestData * LSCoef;

norm(TestEnergy_guess - TestEnergy(:,3)) / sqrt(6831)

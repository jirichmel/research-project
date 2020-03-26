
type=1; %1=train; 2=test; 0=end
count = 1; datacount = 1;
Data=zeros(size(TestLattice,1)+size(TrainLattice,1),21*3);


while type > 0
if type == 1
    AllAtoms=TrainAtoms(TrainAtoms(:,1)==TrainIndMat(count),:);
    AllLattice=TrainLattice(TrainLattice(:,1)==TrainIndMat(count),:);
end

if type == 2
    AllAtoms=TestAtoms(TestAtoms(:,1)==TestIndMat(count),:);
    AllLattice=TestLattice(TestLattice(:,1)==TestIndMat(count),:);
end

AllAtoms(find(AllAtoms(:,2)==0),2)=1;
relaxstep = max(max(AllAtoms(:,2)),1);

for step=1:relaxstep

Atom=AllAtoms(AllAtoms(:,2)==step,:);
Atom=sortrows(Atom,3); %train242 - oxygens first
AtomLattice1=AllLattice(step,3:5);
AtomLattice2=AllLattice(step,6:8);
AtomLattice3=AllLattice(step,9:11);
AtomN=size(Atom,1);
AtomNMetal=AtomN*2/5;
AtomNOxid=AtomN-AtomNMetal;

Distsort=sort(Dist{datacount}(1:AtomNMetal,AtomNMetal+1:AtomN),2);
ratio=zeros(1,8);
oxygen=zeros(AtomN,1);

for u = 1:AtomNMetal
    ratio=zeros(1,8);
    for j=1:min(8,AtomNOxid-1) %AtomN-2
        ratio(j)=Distsort(u,j+1)/Distsort(u,j);
    end
    iddx = find(ratio == max(ratio));
    iddx=max(iddx,3); %iddx=min(iddx,7);
    bound = Distsort(u,iddx);
    oxindx=find(Dist{datacount}(u,:)<=bound & Dist{datacount}(u,:)>0);
    oxygen(oxindx)=oxygen(oxindx)+1;
    
Data(datacount,Atom(u,3)*7+(iddx-8))=Data(datacount,Atom(u,3)*7+(iddx-8))+iddx/AtomNMetal;
Data(datacount,21+Atom(u,3)*7+(iddx-8))=Data(datacount,21+Atom(u,3)*7+(iddx-8))+sum(Distsort(u,1:iddx))/AtomNMetal;
Data(datacount,42+Atom(u,3)*7+(iddx-8))=Data(datacount,42+Atom(u,3)*7+(iddx-8))+norm(Distsort(u,1:iddx),2)^2/AtomNMetal;

%for v=2:5
%    Data(datacount, 63+v) = size(find(oxygen==v),1)/AtomNOxid;
%end

end

datacount=datacount+1;
if mod(datacount, 10000) == 0
    datacount/34000
end

end
    if type == 2 & count == TestNMat
        type = 0
    end

    if type == 2
        count = count + 1;
    end

    if type == 1 & count == TrainNMat
        type = 2; count = 1
    end

    if type == 1
        count = count+1;
    end


end

coef=Data(1:size(TrainEnergy,1),:)\TrainEnergy(:,3);
Energy_guess = Data(size(TrainEnergy,1)+1:size(TestEnergy,1)+size(TrainEnergy,1),:)*coef;
sqrt(norm(Energy_guess-TestEnergy(:,3),2)^2/size(TestEnergy,1)) %0.0665; 0.0533 when ignoring outliers; iddx>=3: 0.0533!

coef2=Data(1:size(TrainEnergy,1),:)\TrainEnergy(:,4);
Bond_guess = Data(size(TrainEnergy,1)+1:size(TestEnergy,1)+size(TrainEnergy,1),:)*coef2;
sqrt(norm(Bond_guess-TestEnergy(:,4),2)^2/size(TestEnergy,1)) %0.3883; iddx>=3: 0,3706!



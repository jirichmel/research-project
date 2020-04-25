
type=1; %1=train; 2=test; 0=end
count = 1; datacount = 1;
Data=zeros(size(TestLattice,1)+size(TrainLattice,1),25);
%Data=zeros(size(TestLattice,1)+size(TrainLattice,1),9);


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
%Dist=zeros(AtomN,AtomN);

%for j=1:AtomNMetal
%    for k=AtomNMetal+1:AtomN
%        Dist(j,k)=norm(Atom(j,4:6)-Atom(k,4:6),2);
%        for x=-1:1
%            for y=-1:1
%                for z=-1:1
%                    Dist(j,k)=min(Dist(j,k), norm(Atom(j,4:6)-Atom(k,4:6)+x*AtomLattice1+y*AtomLattice2+z*AtomLattice3,2));
%                end
%            end
%        end
%    end
%end

Distsort=sort(Dist{datacount}(1:AtomNMetal,AtomNMetal+1:AtomN),2);
%Distsort=sort(Dist);
%ratio=zeros(1,AtomN-2);
ratio=zeros(1,8);
oxygen=zeros(AtomN,1);


for u = 1:AtomNMetal
    ratio=zeros(1,8);
    for j=1:min(8,AtomNOxid-1) %AtomN-2
        ratio(j)=Distsort(u,j+1)/Distsort(u,j);
    end
    iddx = find (ratio == max(ratio));
    bound = Distsort(u,iddx);
    oxindx=find(Dist{datacount}(u,:)<=bound & Dist{datacount}(u,:)>0);
    oxygen(oxindx)=oxygen(oxindx)+1;
    iddx=max(iddx,4); iddx=min(iddx,6);
    
    Data(datacount,Atom(u,3)*7+(iddx-8))=Data(datacount,Atom(u,3)*7+(iddx-8))+1/AtomNMetal;
%Data(datacount,Atom(u,3))=Data(datacount,Atom(u,3))+iddx/AtomNMetal;
%Data(datacount,Atom(u,3)+3)=Data(datacount,Atom(u,3)+3)+sum(Distsort(u,1:iddx))/AtomNMetal;
%Data(datacount,Atom(u,3)+6)=Data(datacount,Atom(u,3)+6)+norm(Distsort(u,1:iddx),2)^2/AtomNMetal;

end

for v=2:5
    Data(datacount, 21+v) = size(find(oxygen==v),1)/AtomNOxid;
end

datacount=datacount+1;
if mod(datacount, 100) == 0
    datacount
end

end
    if type == 2 & count == TestNMat
        type = 0
    end

    if type == 2
        count = count + 1
    end

    if type == 1 & count == TrainNMat
        type = 2; count = 1
    end

    if type == 1
        count = count+1
    end


end

coef=Data(1:size(TrainEnergy,1),:)\TrainEnergy(:,3);
Energy_guess = Data(size(TrainEnergy,1)+1:size(TestEnergy,1)+size(TrainEnergy,1),:)*coef;
sqrt(norm(Energy_guess-TestEnergy(:,3),2)^2/size(TestEnergy,1)) 
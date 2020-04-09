clear;

TrainGen = csvread('../train/relaxation/general.csv', 1);
TrainNMat = size(TrainGen,1);
TrainIndMat = TrainGen(:,1);

TrainLattice = csvread('../train/relaxation/lattice_vector_relaxation.csv',1);
TrainEnergy = csvread('../train/relaxation/energy_relaxation.csv',1);
TrainAtoms = csvread('../train/relaxation/atoms_xyznum_relaxation.csv',1);

TestGen = csvread('../test/relaxation/general.csv', 1);
TestNMat = size(TestGen,1);
TestIndMat = TestGen(:,1);

TestLattice = csvread('../test/relaxation/lattice_vector_relaxation.csv',1);
TestEnergy = csvread('../test/relaxation/energy_relaxation.csv',1);
TestAtoms = csvread('../test/relaxation/atoms_xyznum_relaxation.csv',1);

type=1; %1=train; 2=test; 0=end
count = 1; datacount = 1;
Data=zeros(size(TestLattice,1)+size(TrainLattice,1),21);


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
Dist=zeros(AtomN,AtomN);

for j=1:AtomN
    for k=1:AtomN
        Dist(j,k)=norm(Atom(j,4:6)-Atom(k,4:6),2);
        for x=-1:1
            for y=-1:1
                for z=-1:1
                    Dist(j,k)=min(Dist(j,k), norm(Atom(j,4:6)-Atom(k,4:6)+x*AtomLattice1+y*AtomLattice2+z*AtomLattice3,2));
                end
            end
        end
    end
end


Distsort=sort(Dist);
%ratio=zeros(1,AtomN-2);
ratio=zeros(1,8);

for u = 1:AtomNMetal
    for j=1:8 %AtomN-2
        ratio(j)=Distsort(j+2,u)/Distsort(j+1,u);
    end
    iddx = find (ratio == max(ratio));
    bound = Distsort(iddx+1,u);
%    Data(datacount,Atom(u,3)*3+(iddx-6))=Data(datacount,Atom(u,3)*3+(iddx-6))+1/AtomNMetal;
    Data(datacount,Atom(u,3)*7+(iddx-8))=Data(datacount,Atom(u,3)*7+(iddx-8))+1/AtomNMetal;
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
    
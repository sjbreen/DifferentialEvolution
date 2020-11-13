clear all
%Differential Evolution

%Create synthetic data from a system, eg projectile height 
%y = m1 + m2*t - 0.5*m3*t^2. Our "model" is the set of parameters [m1 m2 m3], ie
%[initial height, initial velocity, gravitational acceleration]. Let
%the "true" model be [20 15 9.8].

%Define a timespan, eg 1 second
t = (0:0.1:1);

%Define a "true" model
truemodel = [20 15 9.8];

%Calculate synthetic data and plot
realdata = truemodel(1) + truemodel(2)*t - 0.5*truemodel(3)*t.^2;

%Define parameter search ranges for [m1 m2 m3]
ParRange.min = [10 5 5];
ParRange.max = [50 40 20];

%Define how many models we want in the population
nsample = 10;

%Initialize a parent population matrix with dimensions nsample x #parameters 
PPop = zeros(nsample,numel(truemodel));

%Calculate random initial parameter values in PPop
for i = 1:size(PPop,1) %For every row
    for j = 1:size(PPop,2) %For every column
        PPop(i,j) = lhsu(ParRange.min(j),ParRange.max(j),1); %lhsu = latin hypercube sampling function
    end
end

%Calculate data for each parent model and store in PModelData with
%dimensions nsample x #time steps. Also Calculate the SSR for each model
%and store in PSSR with dimensions nsample x 1.
PModelData = zeros(size(PPop,1),numel(t));
PSSR = zeros(size(PPop,1),1);
for i = 1:size(PPop,1)
    PModelData(i,:) = PPop(i,1) + PPop(i,2)*t - 0.5*PPop(i,3)*t.^2;
    PSSR(i,1) = sum((realdata - PModelData(i,:)).^2,2);
end

%Calculate the standard deviation of SSR values in PSSR
sigma = std(PSSR);

%Loop counter
counter = 0;

%Main differential evolution loop...
while sigma > 10 %converge at low sigma

    %Initialize a child population, same dimensions as PPop
    CPop = zeros(nsample,numel(truemodel)); 
    
    %Propose each model in Cpop as a combination of PPop models
    for i = 1:size(CPop,1)
        F = 0.6; %Algorithmic parameters
        K = 0.4;
        abc = randperm(size(CPop,1)); %generate a random set of nsample intergers from 1:nsample
        abc = abc(1:3); %take the first three (could be any three)
        
        %Differential evolution equation to find CPop
        CPop(i,:) = PPop(i,:)+F.*(PPop(i,:)-PPop(abc(1),:))+K.*(PPop(abc(2),:)-PPop(abc(3),:));
        
        %If parameters are out of range, set to bounds
        for g=1:numel(ParRange.min);
            if CPop(i,g)>ParRange.max(g) 
                CPop(i,g)=ParRange.max(g);
            elseif CPop(i,g)<ParRange.min(g)
                CPop(i,g)=ParRange.min(g);
            end
        end
    end

    %Calculate data for each child model and store in CModelData with
    %dimensions nsample x #time steps. Also Calculate the SSR for each model
    %and store in CSSR with dimensions nsample x 1.
    CModelData = zeros(size(CPop,1),numel(t));
    CSSR = zeros(size(CPop,1),1);
    for i = 1:nsample
        CModelData(i,:) = CPop(i,1) + CPop(i,2)*t - 0.5*CPop(i,3)*t.^2;
        CSSR(i,1) = sum((realdata - CModelData(i,:)).^2,2);
    end

    %If a child model has a lower SSR than its parent, replace the parent
    %model and parent SSR
    for i = 1:nsample
        if CSSR(i)<PSSR(i) 
            PPop(i,:)=CPop(i,:); 
            PSSR(i)=CSSR(i); 
        end
    end

    %Calculate std. dev. of SSR values of parent population
    sigma = std(PSSR); 
    counter = counter + 1;
    disp(sigma)
    disp(counter)
end

%Find lowest SSR model in PPop, calculate its data, and plot circles over realdata
[minval,I] = min(PSSR);
bestmodel = PPop(I,:);
bestdata = bestmodel(1) + bestmodel(2)*t - 0.5*bestmodel(3)*t.^2;
plot(realdata,'k')
hold on
plot(bestdata,'or')
xlabel('Time [s]');
ylabel('Height [m]');


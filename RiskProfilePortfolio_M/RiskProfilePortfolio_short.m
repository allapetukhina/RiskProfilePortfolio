 %% Loading and data transformation
 
 
tic
clc
clear
%To load data uncomment line below
%load stoxx600NAdata  


%To read data and create initial data uncomment next 29 lines
PRICE         = xlsread('STOXX600 mnemonics names industries P MV 01011998_01012016.xlsx', 'Sheet1');%last column index STOXX NA 600
MV            = xlsread('STOXX600 mnemonics names industries P MV 01011998_01012016.xlsx', 'Sheet3');
INDUSTRY      = xlsread('STOXX600 mnemonics names industries P MV 01011998_01012016.xlsx', 'Sheet2');
[num,txt,raw] = xlsread('STOXX600 mnemonics names industries P MV 01011998_01012016.xlsx', 'Mnemonics');
TICKS         = txt (1, 2:end-1); % first column is a date, last one index STOXX NA 600
alpha         = 0.05;

Date              = x2mdate(PRICE(2:end,1)); % transform data from excel format to Matlab format
Datelong          = x2mdate(PRICE(263:end,1)); % long vector of dates for whole period

Date          = str2num(datestr((Date),'yyyymmdd'));

RET           = [Date, price2ret(PRICE(1:end,2:end))]; %creating covariate and response matrices

RET_Y         = arrayfun(@(x) RET(floor(RET(:,1)/10000) == x, :), unique(floor(RET(:,1)/10000)), 'uniformoutput', false);

RET_YEAR      = cell2struct(RET_Y, 'f1', 2); %returns splited by year

tc            = 1-0.01; %Transaction costs for the portfolio rebalancing

clus_num      = 15; %Define range of clusters' number

kmcdist = {'sqeuclidean', 'cityblock','cosine','correlation'}; % k-means disntaces 

hctdist = {'euclidean', 'seuclidean', 'cityblock','cosine','correlation',...
'minkowski', 'mahalanobis'}; %  distances for
%agglomerative clustering

hctalgo = {'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'}; % agglomerative algorithms 

cmcdist = {'euclidean'; 'seuclidean'; 'sqeuclidean'; 'cityblock'; 'chebychev'; ...
           'mahalanobis'; 'minkowski'; 'cosine'; 'correlation'}; % c-medoids disntaces

nshift    = 0;

save('stoxx600NAdata')
toc % 70-80 sec

%% Calculating of annual risk measures for all stocks in the sample (excluding TS with NA)

tic

for n = 1:length(RET_YEAR)-1
    clear sigma skewn kurt VaR2 Alpha Beta mv Sharpe
    Data                          = RET_YEAR(n).f1(:,2:end);
    [row,col] = find(isnan(Data));%unique(find(isnan(Data)));
    row = unique(row);
    col = unique(col);
    ticks = TICKS;
    ticks(:,col) = [];
    mv = MV(n,:);
    mv( :, col) = [];
    Data(:,any(isnan(Data), 1))   = []; % delete series with NaN
    Date                          = RET_YEAR(n).f1(:,1);

    %1 Sigma
    sigma = std (Data);%Std deviation

    % 2 Skewness
    skewn = skewness(Data); %Skewness

%3 Kurtosis
kurt  = kurtosis(Data);  %Kurtosis
%kurt(:,~any(~isnan(kurt), 1)) = [];
%4 VaR   
VaR2  = prctile(Data, alpha * 100); %Historical VaR

% 5 Expected shortfall
[k,i]    = size(Data);
for m = 1:i

ret_year_m = Data(:, m); 
var2_m = VaR2(1, m);
es(m) = sum(ret_year_m(ret_year_m < var2_m))/(k * alpha);  %Expected shortfall
end

% 6 Beta

[NumSamples, NumSeries]       = size(Data);
NumAssets                     = NumSeries - 1;%1st column -date, end column - index TS. 

StartDate                     = Date(1);
EndDate                       = Date(end);

for i = 1:NumAssets
   

	% Set up separate asset data and design matrices
	TestData   = zeros(NumSamples,1);
	TestDesign = zeros(NumSamples,2);

	TestData(:) = Data(:,i) - 0;% riskless return equals 0
	TestDesign(:,1) = 1.0;
	TestDesign(:,2) = Data(:,end)-0;% riskless return equals 0

	% Estimate CAPM for each asset separately
	[Param, Covar] = ecmmvnrmle(TestData, TestDesign);

	% Estimate ideal standard errors for covariance parameters
	[StdParam, StdCovar] = ecmmvnrstd(TestData, TestDesign, Covar, 'fisher');
	
	% Estimate sample standard errors for model parameters
	StdParam = ecmmvnrstd(TestData, TestDesign, Covar, 'hessian');

	% Set up results for output
	Alpha(i) = Param(1);
	Beta(i) = Param(2);
	Sigma = sqrt(Covar);

	StdAlpha = StdParam(1);
	StdBeta = StdParam(2);
	StdSigma = sqrt(StdCovar);


	
end    


% Sharpe ratios    

Sharpe = sharpe(Data,0);

YEAR(n).DATA    = Data;
YEAR(n).Date    = Date;
YEAR(n).TICK    = ticks;
YEAR(n).SIGMA   = sigma;
YEAR(n).SKEWN   = skewn;
YEAR(n).KURT    = kurt;
YEAR(n).VAR2    = VaR2;
YEAR(n).ES      = es;
YEAR(n).ALPHA   = Alpha;
YEAR(n).BETA    = Beta;
YEAR(n).SHARPE  = Sharpe;
YEAR(n).X       = [YEAR(n).SIGMA(:,1:end-1)', YEAR(n).SKEWN(:,1:end-1)',... 
                  YEAR(n).KURT(:,1:end-1)', YEAR(n).VAR2(:,1:end-1)',...
                  YEAR(n).ES(:,1:end-1)', YEAR(n).BETA'];
YEAR(n).MV      = mv;               
YEAR(n).LOGMV  = log(mv);


nshift = nshift + 1   

end

toc %ca. 8 min

%% Clustering based on absolute values of risk measures
tic
nshift = 0 
for n = 1:length(RET_YEAR)-1 
 
clear X X_norm X_std 
X      = YEAR(n).X; %not modified matrix
X_norm = bsxfun(@rdivide,X,sum(X)); %normalized
X_std  = bsxfun(@rdivide,X,std(X)); %standartized

   
IDX     = [];
IDX_N   = [];
IDX_S   = [];

FCM        = [];
FCM_N      = [];
FCM_S      = [];

HCT     = {};
HCT_N   = {};
HCT_S   = {};

CMC     = [];
CMC_N   = [];
CMC_S   = [];

% Variables for clustering strategies description

%k-means 
KMPAR = [];
kmpar = {};

%Agglomerative
HCTPAR = [];
hctpar = {};

CMCPAR = [];
cmcpar = {};

%k-means
for i = 1:length(kmcdist)
    idx   = [];
    idx_n = [];
    idx_s = [];
    kmpar = {};
    
    for j =  2:clus_num
   
        idx_j    = kmeans(X,j,'Distance', kmcdist(i));
        idx_n_j  = kmeans(X_norm,j,'Distance', kmcdist(i));
        idx_s_j  = kmeans(X_std,j,'Distance', kmcdist(i));
        idx      = [idx, idx_j];
        idx_n    = [idx_n, idx_n_j];
        idx_s    = [idx_s, idx_s_j];
        kmpar{j} = strcat(kmcdist(i),',  ', num2str(j));
    end
    
    IDX     = [IDX, idx];
    IDX_N   = [IDX_N, idx_n];
    IDX_S   = [IDX_S, idx_s];
    KMPAR   = [KMPAR, kmpar];
   
end   
    
    % FUZZY C-means
    for j =  2:clus_num
        [centers,U]    = fcm(X,j);
        [centers_s,US] = fcm(X_std,j);
        [centers_n,UN] = fcm(X_norm,j);
    
        [Umax   fcmidx] = max( U,  [], 1 ); 
        [USmax  fcm_s]  = max( US,  [], 1 ); 
        [UNmax  fcm_n]  = max( UN,  [], 1 ); 
    
        FCM        = [FCM, fcmidx'];
        FCM_N      = [FCM_N, fcm_n'];
        FCM_S      = [FCM_S, fcm_s'];
    
    end
   % FCMDEMO(X)
  

  % Agglomerative hierarchical cluster tree
   % Compute clusters various linkage algos and distances

   
% hctdist = {'euclidean', 'seuclidean', 'cityblock','cosine','correlation',
% 'minkowski', 'mahalanobis', 'cosine', 'correlation'}; %  distances for
% agglomerative clustering
% 
% hctalgo = {'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'}; % agglomerative algorithms
       HCT    = [];
       HCT_N  = [];
       HCT_S  = []; 
       
       HCTPAR  = [];%parapmeters of clustering strategy
  
   for k = 1:length(hctalgo) 
       hct     = {};
       hct_n   = {};
       hct_s   = {};
       
       HCTpar = {};
       
       for i = 1:length(hctdist)

           hct_i     = [];
           hct_n_i   = [];
           hct_s_i   = [];
           hctpar    = [];
           for j =  2:clus_num
               Z_j   = linkage(X,hctalgo(k), hctdist(i));
               Z_n_j = linkage(X_norm,hctalgo(k), hctdist(i));
               Z_s_j = linkage(X_std, hctalgo(k), hctdist(i));   
           
               hct_j   = cluster(Z_j,'maxclust',j);
               hct_n_j = cluster(Z_n_j,'maxclust',j);
               hct_s_j = cluster(Z_s_j,'maxclust',j);
    
               hct_i     = [hct_i, hct_j];
               hct_n_i   = [hct_n_i, hct_n_j];
               hct_s_i   = [hct_s_i, hct_s_j];
               
               hctpar_j   = strcat(hctdist(i),',  ',hctalgo(k), ', ', num2str(j));
               hctpar   = [hctpar, hctpar_j];
           end
           
           hct{i}     = hct_i;
           hct_n{i}   = hct_n_i;
           hct_s{i}   = hct_s_i;
           HCTpar{i}  = hctpar;
       end
    
      HCT     = [HCT; hct];
      HCT_N   = [HCT_N; hct_n];
      HCT_S   = [HCT_S; hct_s]; 
      HCTPAR  = [HCTPAR; HCTpar];
   end

  %C-medoids 
  
  cmcdist = {'euclidean'; 'seuclidean'; 'sqeuclidean'; 'cityblock'; 'chebychev'; ...
             'mahalanobis'; 'minkowski'; 'cosine'; 'correlation'}; % c-medoids disntaces  
    
    for i = 1:length(cmcdist)
        cmc   = [];
        cmc_n = [];
        cmc_s = [];
        for j =  2:clus_num
            [cmc_j,C]   = kmedoids(X, j, 'Distance', cmcdist{i});
            [cmc_n_j,C] = kmedoids(X_norm, j, 'Distance', cmcdist{i});
            [cmc_s_j,C] = kmedoids(X_std, j, 'Distance', cmcdist{i});
            cmcpar{j}   = strcat(cmcdist(i),',  ', num2str(j));
    
            cmc     = [cmc, cmc_j];
            cmc_n   = [cmc_n, cmc_n_j];
            cmc_s   = [cmc_s, cmc_s_j];
        end
        
        CMC     = [CMC, cmc];
        CMC_N   = [CMC_N, cmc_n];
        CMC_S   = [CMC_S, cmc_s];
        CMCPAR  = [CMCPAR, cmcpar];
    end
    
YEAR(n).IDX   = IDX;
YEAR(n).IDX_N = IDX_N;
YEAR(n).IDX_S = IDX_S;

YEAR(n).FCM   = FCM;
YEAR(n).FCM_N = FCM_N;
YEAR(n).FCM_S = FCM_S;

YEAR(n).HCT   = HCT;
YEAR(n).HCT_N = HCT_N;
YEAR(n).HCT_S = HCT_S;

YEAR(n).CMC   = CMC;
YEAR(n).CMC_N = CMC_N;
YEAR(n).CMC_S = CMC_S;

%EVALUATION of a k-means partition 

for i=1:length(kmcdist)
    
YEAR(n).kmc_silhouette{i}         = evalclusters(X, YEAR(n).IDX(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'silhouette');
YEAR(n).kmc_CalinskiHarabasz{i}   = evalclusters(X, YEAR(n).IDX(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'CalinskiHarabasz');
YEAR(n).kmc_DaviesBouldin{i}      = evalclusters(X, YEAR(n).IDX(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'DaviesBouldin');

YEAR(n).kmcs_silhouette{i}        = evalclusters(X_std, YEAR(n).IDX_S(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'silhouette');
YEAR(n).kmcs_CalinskiHarabasz{i}  = evalclusters(X_std, YEAR(n).IDX_S(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'CalinskiHarabasz');
YEAR(n).kmcs_DaviesBouldin{i}     = evalclusters(X_std, YEAR(n).IDX_S(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'DaviesBouldin');

YEAR(n).kmcn_silhouette{i}        = evalclusters(X_norm, YEAR(n).IDX_N(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'silhouette');
YEAR(n).kmcn_CalinskiHarabasz{i}  = evalclusters(X_norm, YEAR(n).IDX_N(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'CalinskiHarabasz');
YEAR(n).kmcn_DaviesBouldin{i}     = evalclusters(X_norm, YEAR(n).IDX_N(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'DaviesBouldin');
end

% evaluation of a FUZZY partition 

YEAR(n).fcm_silhouette         = evalclusters(X, YEAR(n).FCM, 'silhouette');
YEAR(n).fcm_CalinskiHarabasz   = evalclusters(X, YEAR(n).FCM, 'CalinskiHarabasz');
YEAR(n).fcm_DaviesBouldin      = evalclusters(X, YEAR(n).FCM, 'DaviesBouldin');

YEAR(n).fcmn_silhouette        = evalclusters(X_norm, YEAR(n).FCM_N, 'silhouette');
YEAR(n).fcmn_CalinskiHarabasz  = evalclusters(X_norm, YEAR(n).FCM_N, 'CalinskiHarabasz');
YEAR(n).fcmn_DaviesBouldin     = evalclusters(X_norm, YEAR(n).FCM_N, 'DaviesBouldin');

YEAR(n).fcms_silhouette        = evalclusters(X_std, YEAR(n).FCM_S, 'silhouette');
YEAR(n).fcms_CalinskiHarabasz  = evalclusters(X_std, YEAR(n).FCM_S, 'CalinskiHarabasz');
YEAR(n).fcms_DaviesBouldin     = evalclusters(X_std, YEAR(n).FCM_S, 'DaviesBouldin');

% evaluation of a HCT partition 

for i = 1:size(HCT_S,1)*size(HCT_S,2)
YEAR(n).hct_silhouette{i}         = evalclusters(X, YEAR(n).HCT(i), 'silhouette');
YEAR(n).hct_CalinskiHarabasz{i}   = evalclusters(X, YEAR(n).HCT(i), 'CalinskiHarabasz');
YEAR(n).hct_DaviesBouldin{i}      = evalclusters(X, YEAR(n).HCT(i), 'DaviesBouldin');

YEAR(n).hctn_silhouette{i}        = evalclusters(X_norm, YEAR(n).HCT_N(i), 'silhouette');
YEAR(n).hctn_CalinskiHarabasz{i}  = evalclusters(X_norm, YEAR(n).HCT_N(i), 'CalinskiHarabasz');
YEAR(n).hctn_DaviesBouldin{i}     = evalclusters(X_norm, YEAR(n).HCT_N(i), 'DaviesBouldin');

YEAR(n).hcts_silhouette{i}        = evalclusters(X_std, YEAR(n).HCT_S(i), 'silhouette');
YEAR(n).hcts_CalinskiHarabasz{i}  = evalclusters(X_std, YEAR(n).HCT_S(i), 'CalinskiHarabasz');
YEAR(n).hcts_DaviesBouldin{i}     = evalclusters(X_std, YEAR(n).HCT_S(i), 'DaviesBouldin');
end

% evaluation C-medoids partition

for i=1:length(cmcdist)
    
YEAR(n).cmc_silhouette{i}         = evalclusters(X, YEAR(n).CMC(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'silhouette');
YEAR(n).cmc_CalinskiHarabasz{i}   = evalclusters(X, YEAR(n).CMC(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'CalinskiHarabasz');
YEAR(n).cmc_DaviesBouldin{i}      = evalclusters(X, YEAR(n).CMC(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'DaviesBouldin');

YEAR(n).cmcs_silhouette{i}        = evalclusters(X_std, YEAR(n).CMC_S(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'silhouette');
YEAR(n).cmcs_CalinskiHarabasz{i}  = evalclusters(X_std, YEAR(n).CMC_S(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'CalinskiHarabasz');
YEAR(n).cmcs_DaviesBouldin{i}     = evalclusters(X_std, YEAR(n).CMC_S(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'DaviesBouldin');

YEAR(n).cmcn_silhouette{i}        = evalclusters(X_norm, YEAR(n).CMC_N(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'silhouette');
YEAR(n).cmcn_CalinskiHarabasz{i}  = evalclusters(X_norm, YEAR(n).CMC_N(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'CalinskiHarabasz');
YEAR(n).cmcn_DaviesBouldin{i}     = evalclusters(X_norm, YEAR(n).CMC_N(:,1+(clus_num-1)*(i-1):i*(clus_num-1)), 'DaviesBouldin');
end

nshift = nshift + 1 
end
toc %ca 30 min

%% Portfolio construction from maximum sharpe ratios stocks of every cluster
tic
% Choose stocks with maximum Sharpe ratio from every cluster based on
% absolute value of risk measures
nshift = 0 
for n = 1:length(RET_YEAR)-1
Sharpe = YEAR(n).SHARPE(1, 1:end-1);

%K-means
%Max sharpe ratio for every distance

    INDKM       = [];
    INDKM_S     = [];
    INDKM_N     = [];
    
for j = 1:length(kmcdist)
    indkm   = {};
    indkmn  = {}; 
    indkms  = {};
    idx     = YEAR(n).IDX(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
    idxn    = YEAR(n).IDX_N(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
    idxs    = YEAR(n).IDX_S(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
    
    for i = 1:size(idx, 2)
   
       indkm{i}  = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(idx(:,i)==x))))), unique(idx(:,i))', 'uniformoutput', false)); %for k-means clusters 
       indkms{i} = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(idxs(:,i)==x))))), unique(idxs(:,i))', 'uniformoutput', false)); %for Fuzzy C-means clusters
       indkmn{i} = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(idxn(:,i)==x))))), unique(idxn(:,i))', 'uniformoutput', false)); %for  Agglomerative hierarchical clustering 

    end
    
    INDKM       = [INDKM; indkm];
    INDKM_S     = [INDKM_S; indkms];
    INDKM_N     = [INDKM_N; indkmn];
    
   
end

    YEAR(n).INDKM      = INDKM;
    YEAR(n).INDKM_S    = INDKM_S;
    YEAR(n).INDKM_N    = INDKM_N;

%FUZZY clusters
fcmidx  = YEAR(n).FCM;
fcmn    = YEAR(n).FCM_N;
fcms    = YEAR(n).FCM_S;

indfcm   = {};
indfcmn  = {}; 
indfcms  = {};

for i = 1:size(fcmidx, 2)
    
       indfcm{i}     = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(fcmidx(:,i)==x))))), unique(fcmidx(:,i))', 'uniformoutput', false)); %for k-means clusters 
       indfcmn{i}    = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(fcmn(:,i)==x))))), unique(fcmn(:,i))', 'uniformoutput', false)); %for Fuzzy C-means clusters
       indfcms{i}    = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(fcms(:,i)==x))))), unique(fcms(:,i))', 'uniformoutput', false)); %for  Agglomerative hierarchical clustering 

end
 

YEAR(n).INDFCM     = indfcm;
YEAR(n).INDFCM_N   = indfcmn;
YEAR(n).INDFCM_S   = indfcms;


%Agglomerative clustering

    INDHCT       = [];
    INDHCT_S     = [];
    INDHCT_N     = [];

    
for j = 1:size(HCT,1)*size(HCT,2)
    hct    = cell2mat(YEAR(n).HCT(j));
    hctn   = cell2mat(YEAR(n).HCT_N(j));
    hcts   = cell2mat(YEAR(n).HCT_S(j));
    
    indhct = {};
    indhctn = {}; 
    indhcts = {};
 

for i = 1:size(hct, 2)
    
        indhct{i}  = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(hct(:,i)==x))))), unique(hct(:,i))', 'uniformoutput', false)); %for k-means clusters 
        indhcts{i} = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(hcts(:,i)==x))))), unique(hcts(:,i))', 'uniformoutput', false)); %for Fuzzy C-means clusters
        indhctn{i} = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(hctn(:,i)==x))))), unique(hctn(:,i))', 'uniformoutput', false)); %for  Agglomerative hierarchical clustering 
end

    INDHCT         = [INDHCT; indhct];
    INDHCT_S       = [INDHCT_S; indhcts];
    INDHCT_N       = [INDHCT_N; indhctn];
    
end  
YEAR(n).INDHCT     = INDHCT;
YEAR(n).INDHCT_N   = INDHCT_N;
YEAR(n).INDHCT_S   = INDHCT_S;

%C-medoids 

    INDCMC       = [];
    INDCMC_S     = [];
    INDCMC_N     = [];

for j=1:length(cmcdist)
    indcmc = {};
    indcmcn = {}; 
    indcmcs = {};
    cmc     = YEAR(n).CMC(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
    cmcs    = YEAR(n).CMC_N(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
    cmcn    = YEAR(n).CMC_S(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
   for i = 1:size(cmc, 2)
     
       indcmc{i}  = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(cmc(:,i)==x))))), unique(cmc(:,i))', 'uniformoutput', false)); %for k-means clusters 
       indcmcs{i} = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(cmcs(:,i)==x))))), unique(cmcs(:,i))', 'uniformoutput', false)); %for Fuzzy C-means clusters
       indcmcn{i} = cell2mat(arrayfun(@(x)(find(Sharpe==max(Sharpe(1, find(cmcn(:,i)==x))))), unique(cmcn(:,i))', 'uniformoutput', false)); %for  Agglomerative hierarchical clustering 

   end
   
    INDCMC       = [INDCMC; indcmc];
    INDCMC_S     = [INDCMC_S; indcmcs];
    INDCMC_N     = [INDCMC_N; indcmcn];
end

YEAR(n).INDCMC      = INDCMC;
YEAR(n).INDCMC_S    = INDCMC_S;
YEAR(n).INDCMC_N    = INDCMC_N;

nshift = nshift + 1 
end

toc %ca. 30 sec

%% Out of sample 1/n rule portfolios from maximum Sharpe ratio stocks from every cluster based on absolute values 
tic

nshift = 0

 KCAPRETlong     = [];
 KSCAPRETlong    = [];
 KNCAPRETlong    = [];
 
 FCAPRETlong     = [];
 FSCAPRETlong    = [];
 FNCAPRETlong    = [];
 
 HCAPRETlong     = [];
 HSCAPRETlong    = [];
 HNCAPRETlong    = [];
 
 CCAPRETlong     = [];
 CSCAPRETlong    = [];
 CNCAPRETlong    = [];

for n = 2:length(RET_YEAR)-1
       
Data          = YEAR(n).DATA;
Tick          = YEAR(n).TICK;
Tick_old      = YEAR(n-1).TICK;

KCAPoutall  = []; 
KNCAPoutall = []; 
KSCAPoutall = []; 

KCAPRET     = [];
KNCAPRET    = [];
KSCAPRET    = [];
    
indkm_old     = YEAR(n-1).INDKM;
indkmn_old    = YEAR(n-1).INDKM_N;
indkms_old    = YEAR(n-1).INDKM_S;

indfcm_old    = YEAR(n-1).INDFCM;
indfcms_old   = YEAR(n-1).INDFCM_S;
indfcmn_old   = YEAR(n-1).INDFCM_N;

indhct_old    = YEAR(n-1).INDHCT;
indhctn_old   = YEAR(n-1).INDHCT_N;
indhcts_old   = YEAR(n-1).INDHCT_S;

indcmc_old    = YEAR(n-1).INDCMC;
indcmcn_old   = YEAR(n-1).INDCMC_N;
indcmcs_old   = YEAR(n-1).INDCMC_S;


%Portfolios from k-means clusters 
for i = 1:size(indkm_old, 1)*size(indkm_old, 2)

tick_km        = Tick_old(cell2mat(indkm_old(i)));  
indkm          = find(ismember(Tick,tick_km));

tick_kmn       = Tick_old(cell2mat(indkmn_old(i)));  
indkmn         = find(ismember(Tick,tick_kmn));

tick_kms       = Tick_old(cell2mat(indkms_old(i)));  
indkms         = find(ismember(Tick,tick_kms));

clear kcap kcapp kcappp  

KCAPout  = []; 
kcappp   = [];
kcap{1}  = 1; 
kcapp(1) = 1;

KSCAPout  = []; 
kscappp   = [];
kscap{1}  = 1; 
kscapp(1) = 1;

KNCAPout  = []; 
kncappp   = [];
kncap{1}  = 1; 
kncapp(1) = 1;


for l = 1:size(Data,1)  
    
     kcap{l}   = (repmat(sum(cell2mat(kcap(l)))/length(indkm),1,length(indkm)));
     kcap{l+1} = sum(cell2mat(kcap(l)).*(1 + Data(l,(indkm))));
     kcapp     = kcap{l+1};
     kcappp    = [kcappp,kcapp];  
     
     kncap{l}   = (repmat(sum(cell2mat(kncap(l)))/length(indkmn),1,length(indkmn)));
     kncap{l+1} = sum(cell2mat(kncap(l)).*(1 + Data(l,(indkmn))));
     kncapp     = kncap{l+1};
     kncappp    = [kncappp,kncapp];  
     
     kscap{l}   = (repmat(sum(cell2mat(kscap(l)))/length(indkms),1,length(indkms)));
     kscap{l+1} = sum(cell2mat(kscap(l)).*(1 + Data(l,(indkms))));
     kscapp     = kscap{l+1};
     kscappp    = [kscappp,kscapp];  
end   

     KCAPout       = [1, kcappp]; 
     KCAPoutall    = [KCAPoutall; KCAPout]; 
     KCAPRET       = [KCAPRET; price2ret(KCAPout,[], 'Periodic')];
     
     KNCAPout      = [1, kncappp];
     KNCAPoutall   = [KNCAPoutall; KNCAPout]; 
     KNCAPRET      = [KNCAPRET; price2ret(KNCAPout,[], 'Periodic')];
     
     KSCAPout      = [1,kscappp]; 
     KSCAPRET      = [KSCAPRET; price2ret(KSCAPout,[], 'Periodic')];
     KSCAPoutall   = [KSCAPoutall; KSCAPout]; 

end
     
 KCAPRETlong     = [KCAPRETlong, KCAPRET];
 KSCAPRETlong    = [KSCAPRETlong, KSCAPRET];
 KNCAPRETlong    = [KNCAPRETlong, KNCAPRET];

 %Portfolios from FUZZY clusters with 
 
FCAPoutall  = []; 
FNCAPoutall = []; 
FSCAPoutall = []; 


FCAPRET     = [];
FNCAPRET    = [];
FSCAPRET    = [];
 for i = 1:size(indfcm_old, 1)*size(indfcm_old, 2)

clear fcap fcapp fcappp 

FCAPout  = []; 
fcappp   = [];
fcap{1}  = 1; 
fcapp(1) = 1;

FSCAPout  = []; 
fscappp   = [];
fscap{1}  = 1; 
fscapp(1) = 1;

FNCAPout  = []; 
fncappp   = [];
fncap{1}  = 1; 
fncapp(1) = 1;

tick_fcm       = Tick_old(cell2mat(indfcm_old(i)));  
indfcm         = find(ismember(Tick,tick_fcm));

tick_fcmn      = Tick_old(cell2mat(indfcmn_old(i)));  
indfcmn        = find(ismember(Tick,tick_fcmn));

tick_fcms      = Tick_old(cell2mat(indfcms_old(i)));  
indfcms        = find(ismember(Tick,tick_fcms));


for l = 1:size(Data,1)  
    
     fcap{l}   = (repmat(sum(cell2mat(fcap(l)))/length(indfcm),1,length(indfcm)));
     fcap{l+1} = sum(cell2mat(fcap(l)).*(1 + Data(l,(indfcm))));
     fcapp     = fcap{l+1};
     fcappp    = [fcappp,fcapp];  
     
     fncap{l}   = (repmat(sum(cell2mat(fncap(l)))/length(indfcmn),1,length(indfcmn)));
     fncap{l+1} = sum(cell2mat(fncap(l)).*(1 + Data(l,(indfcmn))));
     fncapp     = fncap{l+1};
     fncappp    = [fncappp,fncapp];  
     
     fscap{l}   = (repmat(sum(cell2mat(fscap(l)))/length(indfcms),1,length(indfcms)));
     fscap{l+1} = sum(cell2mat(fscap(l)).*(1 + Data(l,(indfcms))));
     fscapp     = fscap{l+1};
     fscappp    = [fscappp,fscapp];  
end   

     FCAPout    = [1, fcappp]; 
     FCAPoutall = [FCAPoutall; FCAPout]; 
     FCAPRET    = [FCAPRET; price2ret(FCAPout,[], 'Periodic')];
     
     FNCAPout    = [1, fncappp];
     FNCAPoutall = [FNCAPoutall; FNCAPout]; 
     FNCAPRET    = [FNCAPRET; price2ret(FNCAPout,[], 'Periodic')];
     

     FSCAPout    = [1, fscappp]; 
     FSCAPRET    = [FSCAPRET; price2ret(FSCAPout,[], 'Periodic')];
     FSCAPoutall = [FSCAPoutall; FSCAPout]; 

end
     
 FCAPRETlong     = [FCAPRETlong, FCAPRET];
 FSCAPRETlong    = [FSCAPRETlong, FSCAPRET];
 FNCAPRETlong    = [FNCAPRETlong, FNCAPRET];
    
%Portfolios from hct clusters with 

HCAPoutall  = []; 
HNCAPoutall = []; 
HSCAPoutall = []; 

HCAPRET     = [];
HNCAPRET    = [];
HSCAPRET    = [];
     
 for i = 1:size(indhct_old, 1)*size(indhct_old, 2)

tick_hct        = Tick_old(cell2mat(indhct_old(i)));  
indhct          = find(ismember(Tick,tick_hct));

tick_hctn       = Tick_old(cell2mat(indhctn_old(i)));  
indhctn         = find(ismember(Tick,tick_hctn));

tick_hcts       = Tick_old(cell2mat(indhcts_old(i)));  
indhcts         = find(ismember(Tick,tick_hcts));

clear hcap hcapp hcappp 

HCAPout  = []; 
hcappp   = [];
hcap{1}  = 1; 
hcapp(1) = 1;

HSCAPout  = []; 
hscappp   = [];
hscap{1}  = 1; 
hscapp(1) = 1;

HNCAPout  = []; 
hncappp   = [];
hncap{1}  = 1; 
hncapp(1) = 1;

for l = 1:size(Data,1)  
    
     hcap{l}   = (repmat(sum(cell2mat(hcap(l)))/length(indhct),1,length(indhct)));
     hcap{l+1} = sum(cell2mat(hcap(l)).*(1 + Data(l,(indhct))));
     hcapp     = hcap{l+1};
     hcappp    = [hcappp,hcapp];  
     
     hncap{l}   = (repmat(sum(cell2mat(hncap(l)))/length(indhctn),1,length(indhctn)));
     hncap{l+1} = sum(cell2mat(hncap(l)).*(1 + Data(l,(indhctn))));
     hncapp     = hncap{l+1};
     hncappp    = [hncappp,hncapp];  
     
     hscap{l}   = (repmat(sum(cell2mat(hscap(l)))/length(indhcts),1,length(indhcts)));
     hscap{l+1} = sum(cell2mat(hscap(l)).*(1 + Data(l,(indhcts))));
     hscapp     = hscap{l+1};
     hscappp    = [hscappp,hscapp];  
end   

     HCAPout    = [1, hcappp]; 
     HCAPoutall = [HCAPoutall; HCAPout]; 
     HCAPRET    = [HCAPRET; price2ret(HCAPout,[], 'Periodic')];
     
     HNCAPout    = [1, hncappp];
     HNCAPoutall = [HNCAPoutall; HNCAPout]; 
     HNCAPRET    = [HNCAPRET; price2ret(HNCAPout,[], 'Periodic')];

     HSCAPout    = [1, hscappp]; 
     HSCAPRET    = [HSCAPRET; price2ret(HSCAPout,[], 'Periodic')];
     HSCAPoutall = [HSCAPoutall; HSCAPout]; 

end
     
 HCAPRETlong     = [HCAPRETlong, HCAPRET];
 HSCAPRETlong    = [HSCAPRETlong, HSCAPRET];
 HNCAPRETlong    = [HNCAPRETlong, HNCAPRET];
 
 % Portfolios from C-medoids clusters
 
CCAPoutall  = []; 
CNCAPoutall = []; 
CSCAPoutall = []; 

CCAPRET     = [];
CNCAPRET    = [];
CSCAPRET    = [];

for i = 1:size(indcmc_old, 1)*size(indcmc_old, 2)
    
tick_cmc       = Tick_old(cell2mat(indcmc_old(i)));  
indcmc         = find(ismember(Tick,tick_cmc));

tick_cmcn       = Tick_old(cell2mat(indcmcn_old(i)));  
indcmcn         = find(ismember(Tick,tick_cmcn));

tick_cmcs       = Tick_old(cell2mat(indcmcs_old(i)));  
indcmcs         = find(ismember(Tick,tick_cmcs));

clear ccap ccapp ccappp  

CCAPout  = []; 
ccappp   = [];
ccap{1}  = 1; 
ccapp(1) = 1;

CSCAPout  = []; 
cscappp   = [];
cscap{1}  = 1; 
cscapp(1) = 1;

CNCAPout  = []; 
cncappp   = [];
cncap{1}  = 1; 
cncapp(1) = 1;


     for l = 1:size(Data,1)  
    
     ccap{l}   = (repmat(sum(cell2mat(ccap(l)))/length(indcmc),1,length(indcmc)));
     ccap{l+1} = sum(cell2mat(ccap(l)).*(1 + Data(l,(indcmc))));
     ccapp     = ccap{l+1};
     ccappp    = [ccappp,ccapp];  
     
     cncap{l}   = (repmat(sum(cell2mat(cncap(l)))/length(indcmcn),1,length(indcmcn)));
     cncap{l+1} = sum(cell2mat(cncap(l)).*(1 + Data(l,(indcmcn))));
     cncapp     = cncap{l+1};
     cncappp    = [cncappp,cncapp];  
     
     cscap{l}   = (repmat(sum(cell2mat(cscap(l)))/length(indcmcs),1,length(indcmcs)));
     cscap{l+1} = sum(cell2mat(cscap(l)).*(1 + Data(l,(indcmcs))));
     cscapp     = cscap{l+1};
     cscappp    = [cscappp,cscapp];  
     end   

     CCAPout    = [1,ccappp]; 
     CCAPoutall = [CCAPoutall; CCAPout]; 
     CCAPRET    = [CCAPRET; price2ret(CCAPout,[], 'Periodic')];
     
     CNCAPout    = [1,cncappp];
     CNCAPoutall   = [CNCAPoutall; CNCAPout]; 
     CNCAPRET    = [CNCAPRET; price2ret(CNCAPout,[], 'Periodic')];
     
     CSCAPout    = [1,cscappp]; 
     CSCAPRET    = [CSCAPRET; price2ret(CSCAPout,[], 'Periodic')];
     CSCAPoutall = [CSCAPoutall; CSCAPout]; 

end
     
 CCAPRETlong     = [CCAPRETlong, CCAPRET];
 CSCAPRETlong    = [CSCAPRETlong, CSCAPRET];
 CNCAPRETlong    = [CNCAPRETlong, CNCAPRET];


YEAR(n).KCAPout     = KCAPoutall;
YEAR(n).FCAPout     = FCAPoutall;
YEAR(n).HCAPout     = HCAPoutall;
YEAR(n).CCAPout     = CCAPoutall;


YEAR(n).KSCAPout  = KSCAPoutall;
YEAR(n).FSCAPout  = FSCAPoutall;
YEAR(n).HSCAPout  = HSCAPoutall;
YEAR(n).CSCAPout  = CSCAPoutall;

YEAR(n).KNCAPout  = KNCAPoutall;
YEAR(n).FNCAPout  = FNCAPoutall;
YEAR(n).HNCAPout  = HNCAPoutall;
YEAR(n).CNCAPout  = CNCAPoutall;

YEAR(n).KCAPRET   = KCAPRET;
YEAR(n).FCAPRET   = FCAPRET;
YEAR(n).HCAPRET   = HCAPRET;
YEAR(n).CCAPRET   = CCAPRET;

YEAR(n).KSCAPRET  = KSCAPRET;
YEAR(n).FSCAPRET  = FSCAPRET;
YEAR(n).HSCAPRET  = HSCAPRET;
YEAR(n).CSCAPRET  = CSCAPRET;

YEAR(n).KNCAPRET  = KNCAPRET;
YEAR(n).FNCAPRET  = FNCAPRET;
YEAR(n).HNCAPRET  = HNCAPRET;
YEAR(n).CNCAPRET  = CNCAPRET;

nshift = nshift + 1
end
toc %ca 7 min

%% Construction of long 1/n portfolios' cumulative returns 
tic
%parpool(12)
% k-means
    KCAPoutlong = [];
    KSCAPoutlong = [];    
    KNCAPoutlong = [];    
parfor j = 1:size(KCAPRETlong,1)
    IND = [];
    IND = KCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    

    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
   % indcaplast  = INDCAP(i+1);
    
    INDCAPIT    = [INDCAP];    
    
       end
   KCAPoutlong = [KCAPoutlong; INDCAPIT];
end   
parfor j = 1:size(KCAPRETlong,1)
    IND = [];
    IND = KNCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));

    INDCAPIT  = [INDCAP];    
    
       end
   KNCAPoutlong = [KNCAPoutlong; INDCAPIT];
end   
parfor j = 1:size(KCAPRETlong,1)
    IND = [];
    IND = KSCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
  
    INDCAPIT  = [INDCAP];    
    
    end
    KSCAPoutlong = [KSCAPoutlong; INDCAPIT];
end   
% C-medoids
    CCAPoutlong = [];
    CSCAPoutlong = [];    
    CNCAPoutlong = [];
parfor j = 1:size(CCAPRETlong,1)
    IND = [];
    IND = CCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
   % indcaplast  = INDCAP(i+1);
    
    INDCAPIT    = [INDCAP];    
    
       end
   CCAPoutlong = [CCAPoutlong; INDCAPIT];
end   
parfor j = 1:size(CCAPRETlong,1)
    IND = [];
    IND = CNCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));

    INDCAPIT    = [INDCAP];    
    
       end
   CNCAPoutlong = [CNCAPoutlong; INDCAPIT];
end   
parfor j = 1:size(CCAPRETlong,1)
    IND = [];
    IND = CSCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
  
    INDCAPIT  = [INDCAP];    
    
       end
   CSCAPoutlong = [CSCAPoutlong; INDCAPIT];
end   

% Agglomerative clustering
    HCAPoutlong = [];
    HSCAPoutlong = [];    
    HNCAPoutlong = [];
    
parfor j = 1:size(HCAPRETlong,1)
    IND = [];
    IND = HCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
   % indcaplast  = INDCAP(i+1);
    
    INDCAPIT    = [INDCAP];    
   
       end
   HCAPoutlong = [HCAPoutlong; INDCAPIT];
end   
parfor j = 1:size(HCAPRETlong,1)
       IND = [];
       IND = HNCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));

    INDCAPIT    = [INDCAP];    
    
       end
   HNCAPoutlong = [HNCAPoutlong; INDCAPIT];
end   
parfor j = 1:size(HCAPRETlong,1)
    IND = [];
    IND = HSCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
    INDCAPIT  = [INDCAP];    
    
       end
   HSCAPoutlong = [HSCAPoutlong; INDCAPIT];
end    

% FUZZY clustering


    FCAPoutlong = [];
    FSCAPoutlong = [];    
    FNCAPoutlong = [];
    
parfor j = 1:size(FCAPRETlong,1)
    IND = [];
    IND = FCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
   % indcaplast  = INDCAP(i+1);
    
    INDCAPIT    = [INDCAP];    
    
       end
   FCAPoutlong = [FCAPoutlong; INDCAPIT];
end   
parfor j = 1:size(FCAPRETlong,1)
    IND = [];
    IND = FNCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];

    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
    INDCAPIT    = [INDCAP];    
    
    end
   FNCAPoutlong = [FNCAPoutlong; INDCAPIT];
end   
parfor j = 1:size(FCAPRETlong,1)
    IND = [];
    IND = FSCAPRETlong(j,:);
    INDCAP  = [];
    INDCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDCAP(1) = 1;       
    
    INDCAP(i) = INDCAP(i-1)*(1+IND(i));
  
    INDCAPIT  = [INDCAP];    
    
       end
   FSCAPoutlong = [FSCAPoutlong; INDCAPIT];
end   


toc %ca 34 min

%% Construction of long 1/n portfolios' cumulative returns with TC
nshift = 0
% k-means
tic
    KCAPTCoutlong  = YEAR(2).KCAPout(:,2:end);
    KSCAPTCoutlong = YEAR(2).KSCAPout(:,2:end);     
    KNCAPTCoutlong = YEAR(2).KNCAPout(:,2:end);
    
    HCAPTCoutlong  = YEAR(2).HCAPout(:,2:end);
    HSCAPTCoutlong = YEAR(2).HSCAPout(:,2:end);   
    HNCAPTCoutlong = YEAR(2).HNCAPout(:,2:end);
    
    CCAPTCoutlong  = YEAR(2).CCAPout(:,2:end);
    CSCAPTCoutlong = YEAR(2).CSCAPout(:,2:end);    
    CNCAPTCoutlong = YEAR(2).CNCAPout(:,2:end);
    
    FCAPTCoutlong  = YEAR(2).FCAPout(:,2:end);
    FSCAPTCoutlong = YEAR(2).FSCAPout(:,2:end);   
    FNCAPTCoutlong = YEAR(2).FNCAPout(:,2:end);
   
for n = 3:length(RET_YEAR)-1
    %k-means
    INDCAPTCIT    = [];
    for j = 1:size(KCAPRETlong,1)
            IND = [];
            IND = YEAR(n).KCAPRET(j,:); 
            INDCAPTC    = []; 
  
            for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
          
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    KCAPTCoutlong = [KCAPTCoutlong, INDCAPTCIT(:,2:end)];

    INDCAPTCIT    = [];      
    for j = 1:size(KNCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).KNCAPRET(j,:); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KNCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    KNCAPTCoutlong = [KNCAPTCoutlong, INDCAPTCIT(:,2:end)];  
    
    INDCAPTCIT    = []; 
    for j = 1:size(KSCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).KSCAPRET(j,:); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KSCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    KSCAPTCoutlong = [KSCAPTCoutlong, INDCAPTCIT(:,2:end)];  
     
 
% C-medoids

    INDCAPTCIT    = [];
    for j = 1:size(CCAPRETlong,1)
           IND = [];
           IND = YEAR(n).CCAPRET(j,:); 
           INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
                
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    CCAPTCoutlong = [CCAPTCoutlong, INDCAPTCIT(:,2:end)];
   
    INDCAPTCIT    = [];      
    for j = 1:size(CNCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).CNCAPRET(j,:); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CNCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    CNCAPTCoutlong = [CNCAPTCoutlong, INDCAPTCIT(:,2:end)];  
    
    INDCAPTCIT    = []; 
    for j = 1:size(CSCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).CSCAPRET(j,:); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CSCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    CSCAPTCoutlong = [CSCAPTCoutlong, INDCAPTCIT(:,2:end)];   
 
% Agglomerative clustering
    INDCAPTCIT    = [];
    for j = 1:size(HCAPRETlong,1)
           IND = [];
           IND = YEAR(n).HCAPRET(j,:); 
           INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
                
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    HCAPTCoutlong = [HCAPTCoutlong, INDCAPTCIT(:,2:end)];
   
    INDCAPTCIT    = [];      
    for j = 1:size(HNCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).HNCAPRET(j,:); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HNCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    HNCAPTCoutlong = [HNCAPTCoutlong, INDCAPTCIT(:,2:end)];  
    
    INDCAPTCIT    = []; 
    for j = 1:size(HSCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).HSCAPRET(j,:); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HSCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    HSCAPTCoutlong = [HSCAPTCoutlong, INDCAPTCIT(:,2:end)];   
 

% FUZZY clustering
    INDCAPTCIT    = [];
    for j = 1:size(FCAPRETlong,1)
           IND = [];
           IND = YEAR(n).FCAPRET(j,:); 
           INDCAPTC    = []; 
           
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
                
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    FCAPTCoutlong = [FCAPTCoutlong, INDCAPTCIT(:,2:end)];
   
    INDCAPTCIT    = [];      
    for j = 1:size(FNCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).FNCAPRET(j,:); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FNCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    FNCAPTCoutlong = [FNCAPTCoutlong, INDCAPTCIT(:,2:end)];  
    
    INDCAPTCIT    = []; 
    for j = 1:size(FSCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).FSCAPRET(j,:); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FSCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    FSCAPTCoutlong = [FSCAPTCoutlong, INDCAPTCIT(:,2:end)];  
 

nshift = nshift + 1 
end 

toc %ca 

%% Out of sample Mean-variance portfolios from maximum Sharpe ratio stocks from every cluster based on absolute values 
tic
nshift = 0

 KMVCAPRETlong     = [];
 KSMVCAPRETlong    = [];
 KNMVCAPRETlong    = [];
 
 FMVCAPRETlong     = [];
 FSMVCAPRETlong    = [];
 FNMVCAPRETlong    = [];
 
 HMVCAPRETlong     = [];
 HSMVCAPRETlong    = [];
 HNMVCAPRETlong    = [];
 
 CMVCAPRETlong     = [];
 CSMVCAPRETlong    = [];
 CNMVCAPRETlong    = [];

for n = 2:length(RET_YEAR)-1
       
Data          = YEAR(n).DATA;
Data_old      = YEAR(n-1).DATA;
Tick          = YEAR(n).TICK;
Tick_old      = YEAR(n-1).TICK;

KMVCAPoutall  = []; 
KNMVCAPoutall = []; 
KSMVCAPoutall = []; 

FMVCAPoutall  = []; 
FNMVCAPoutall = []; 
FSMVCAPoutall = []; 

HMVCAPoutall  = []; 
HNMVCAPoutall = []; 
HSMVCAPoutall = []; 

CMVCAPoutall    = []; 
CNMVCAPoutall   = []; 
CSMVCAPoutall   = []; 

KMVCAPRET     = [];
KNMVCAPRET    = [];
KSMVCAPRET    = [];

FMVCAPRET     = [];
FNMVCAPRET    = [];
FSMVCAPRET    = [];

CMVCAPRET     = [];
CNMVCAPRET    = [];
CSMVCAPRET    = [];

HMVCAPRET     = [];
HNMVCAPRET    = [];
HSMVCAPRET    = [];
 
indkm_old     = YEAR(n-1).INDKM;
indkmn_old    = YEAR(n-1).INDKM_N;
indkms_old    = YEAR(n-1).INDKM_S;

indfcm_old    = YEAR(n-1).INDFCM;
indfcms_old   = YEAR(n-1).INDFCM_S;
indfcmn_old   = YEAR(n-1).INDFCM_N;

indhct_old    = YEAR(n-1).INDHCT;
indhctn_old   = YEAR(n-1).INDHCT_N;
indhcts_old   = YEAR(n-1).INDHCT_S;

indcmc_old    = YEAR(n-1).INDCMC;
indcmcn_old   = YEAR(n-1).INDCMC_N;
indcmcs_old   = YEAR(n-1).INDCMC_S;

num_dig       = 4;
TargRet       = 0.8;
options       = optimset('Algorithm','active-set','MaxFunEvals',100000);


% Portfolios from k-means clusters with 

for i = 1:size(indkm_old, 1)*size(indkm_old, 2)

tick_km       = Tick_old(cell2mat(indkm_old(i)));  
indkm         = find(ismember(Tick,tick_km));

tick_kmn       = Tick_old(cell2mat(indkmn_old(i)));  
indkmn         = find(ismember(Tick,tick_kmn));

tick_kms       = Tick_old(cell2mat(indkms_old(i)));  
indkms         = find(ismember(Tick,tick_kms));

%clear kcap kcapp kcappp  kncap kncapp kncappp kscap kscapp kscappp

KMVCAPout  = []; 
kcappp   = [];
kcap{1}  = 1; 
kcapp(1) = 1;

KSMVCAPout  = []; 
kscappp   = [];
kscap{1}  = 1; 
kscapp(1) = 1;

KNMVCAPout  = []; 
kncappp   = [];
kncap{1}  = 1; 
kncapp(1) = 1;


    parfor l = 1:size(Data,1)  
     
        XI               = Data_old(:,(cell2mat(indkm_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kwgt             = round(kwgt.*(10^num_dig))./(10^num_dig);
        Kwgt{i}          = kwgt;
     
        kcap{l}          = sum(cell2mat(kcap(l)));    
% %portfolio appreciation
        kcap{l+1}        = sum(cell2mat(kcap(l)).*kwgt'.*(1 + Data(l,(indkm))'));
        kcapp            = kcap{l+1};    
        kcappp           = [kcappp,kcapp];
        
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        XI               = Data_old(:,(cell2mat(indkmn_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [knwgt,iVaR]     = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        knwgt            = round(knwgt.*(10^num_dig))./(10^num_dig);
        Knwgt{i}         = knwgt;
     
        kncap{l}         = sum(cell2mat(kncap(l)));    
% %portfolio appreciation
        kncap{l+1}       = sum(cell2mat(kncap(l)).*knwgt'.*(1 + Data(l,(indkmn))'));
        kncapp           = kncap{l+1};    
        kncappp          = [kncappp,kncapp];
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        XI               = Data_old(:,(cell2mat(indkms_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]     = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt            = round(kswgt.*(10^num_dig))./(10^num_dig);
        Kswgt{i}         = kswgt;
     
        kscap{l}         = sum(cell2mat(kscap(l)));    
% %portfolio appreciation
        kscap{l+1}       = sum(cell2mat(kscap(l)).*kswgt'.*(1 + Data(l,(indkms))'));
        kscapp           = kscap{l+1};    
        kscappp          = [kscappp,kscapp];
    end
    
    KMVCAPout       = [1,kcappp]; 
    KMVCAPoutall    = [KMVCAPoutall; KMVCAPout]; 
    KMVCAPRET       = [KMVCAPRET; price2ret(KMVCAPout,[], 'Periodic')]; 
    
    KSMVCAPout      = [1,kscappp]; 
    KSMVCAPoutall   = [KSMVCAPoutall; KSMVCAPout]; 
    KSMVCAPRET      = [KSMVCAPRET; price2ret(KSMVCAPout,[], 'Periodic')]; 
    
    KNMVCAPout      = [1,kncappp]; 
    KNMVCAPoutall   = [KNMVCAPoutall; KNMVCAPout]; 
    KNMVCAPRET      = [KNMVCAPRET; price2ret(KNMVCAPout,[], 'Periodic')]; 

end
     
 KMVCAPRETlong     = [KMVCAPRETlong, KMVCAPRET];
 KSMVCAPRETlong    = [KSMVCAPRETlong, KSMVCAPRET];
 KNMVCAPRETlong    = [KNMVCAPRETlong, KNMVCAPRET];

 % Portfolios from FUZZY clusters with 
for i = 1:size(indfcm_old, 1)*size(indfcm_old, 2)


tick_fcm       = Tick_old(cell2mat(indfcm_old(i)));  
indfcm         = find(ismember(Tick,tick_fcm));

tick_fcmn       = Tick_old(cell2mat(indfcmn_old(i)));  
indfcmn         = find(ismember(Tick,tick_fcmn));

tick_fcms       = Tick_old(cell2mat(indfcms_old(i)));  
indfcms         = find(ismember(Tick,tick_fcms));

%clear fcap fcapp fcappp  fncap fncapp fncappp fscap fscapp fscappp

FMVCAPout  = []; 
fcappp   = [];
fcap{1}  = 1; 
fcapp(1) = 1;

FSMVCAPout  = []; 
fscappp   = [];
fscap{1}  = 1; 
fscapp(1) = 1;

FNMVCAPout  = []; 
fncappp   = [];
fncap{1}  = 1; 
fncapp(1) = 1;


    for l = 1:size(Data,1)  
     
        XI               = Data_old(:,(cell2mat(indfcm_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kwgt             = round(kwgt.*(10^num_dig))./(10^num_dig);
        Kwgt{i}          = kwgt;
     
        fcap{l}          = sum(cell2mat(fcap(l)));    
% %portfolio appreciation
        fcap{l+1}        = sum(cell2mat(fcap(l)).*kwgt'.*(1 + Data(l,(indfcm))'));
        fcapp            = fcap{l+1};    
        fcappp           = [fcappp,fcapp];
        
        
%%%%%%%%%%%%%
        XI               = Data_old(:,(cell2mat(indfcmn_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [knwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        knwgt             = round(knwgt.*(10^num_dig))./(10^num_dig);
        Knwgt{i}          = knwgt;
     
        fncap{l}          = sum(cell2mat(fncap(l)));    
% %portfolio appreciation
        fncap{l+1}        = sum(cell2mat(fncap(l)).*knwgt'.*(1 + Data(l,(indfcmn))'));
        fncapp            = fncap{l+1};    
        fncappp           = [fncappp,fncapp];
 %%%%%%%%%%%%%%%%%%       
        XI               = Data_old(:,(cell2mat(indfcms_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt             = round(kswgt.*(10^num_dig))./(10^num_dig);
        Kswgt{i}          = kswgt;
     
        fscap{l}          = sum(cell2mat(fscap(l)));    
% %portfolio appreciation
        fscap{l+1}        = sum(cell2mat(fscap(l)).*kswgt'.*(1 + Data(l,(indfcms))'));
        fscapp            = fscap{l+1};    
        fscappp            = [fscappp,fscapp];
    end
    
    FMVCAPout       = [1,fcappp]; 
    FMVCAPoutall    = [FMVCAPoutall; FMVCAPout]; 
    FMVCAPRET       = [FMVCAPRET; price2ret(FMVCAPout,[], 'Periodic')]; 
    
    FSMVCAPout      = [1,fscappp]; 
    FSMVCAPoutall   = [FSMVCAPoutall; FSMVCAPout]; 
    FSMVCAPRET      = [FSMVCAPRET; price2ret(FSMVCAPout,[], 'Periodic')]; 
    
    FNMVCAPout      = [1,fncappp]; 
    FNMVCAPoutall   = [FNMVCAPoutall; FNMVCAPout]; 
    FNMVCAPRET       = [FNMVCAPRET; price2ret(FNMVCAPout,[], 'Periodic')]; 

end
     
 FMVCAPRETlong     = [FMVCAPRETlong, FMVCAPRET];
 FSMVCAPRETlong    = [FSMVCAPRETlong, FSMVCAPRET];
 FNMVCAPRETlong    = [FNMVCAPRETlong, FNMVCAPRET];
% Portfolios from hct clusters 
 
for i = 1:size(indhct_old, 1)*size(indhct_old, 2)


tick_hct       = Tick_old(cell2mat(indhct_old(i)));  
indhct         = find(ismember(Tick,tick_hct));

tick_hctn       = Tick_old(cell2mat(indhctn_old(i)));  
indhctn         = find(ismember(Tick,tick_hctn));

tick_hcts       = Tick_old(cell2mat(indhcts_old(i)));  
indhcts         = find(ismember(Tick,tick_hcts));

%clear hcap hcapp hcappp  hncap hncapp hncappp hscap hscapp hscappp

HMVCAPout  = []; 
hcappp   = [];
hcap{1}  = 1; 
hcapp(1) = 1;

HSMVCAPout  = []; 
hscappp   = [];
hscap{1}  = 1; 
hscapp(1) = 1;

HNMVCAPout  = []; 
hncappp   = [];
hncap{1}  = 1; 
hncapp(1) = 1;


    for l = 1:size(Data,1)  
     
        XI               = Data_old(:,(cell2mat(indhct_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kwgt             = round(kwgt.*(10^num_dig))./(10^num_dig);
        Kwgt{i}          = kwgt;
     
        hcap{l}          = sum(cell2mat(hcap(l)));    
% %portfolio appreciation
        hcap{l+1}        = sum(cell2mat(hcap(l)).*kwgt'.*(1 + Data(l,(indhct))'));
        hcapp            = hcap{l+1};    
        hcappp           = [hcappp,hcapp];
        
        3
%%%%%%%%%%%%%
        XI               = Data_old(:,(cell2mat(indhctn_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [knwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        knwgt             = round(knwgt.*(10^num_dig))./(10^num_dig);
        Knwgt{i}          = knwgt;
     
        hncap{l}          = sum(cell2mat(hncap(l)));    
% %portfolio appreciation
        hncap{l+1}        = sum(cell2mat(hncap(l)).*knwgt'.*(1 + Data(l,(indhctn))'));
        hncapp            = hncap{l+1};    
        hncappp           = [hncappp,hncapp];
 %%%%%%%%%%%%%%%%%%       
        XI               = Data_old(:,(cell2mat(indhcts_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt             = round(kswgt.*(10^num_dig))./(10^num_dig);
        Kswgt{i}          = kswgt;
     
        hscap{l}          = sum(cell2mat(hscap(l)));    
% %portfolio appreciation
        hscap{l+1}        = sum(cell2mat(hscap(l)).*kswgt'.*(1 + Data(l,(indhcts))'));
        hscapp            = hscap{l+1};    
        hscappp            = [hscappp,hscapp];
    end
    
    HMVCAPout       = [1,hcappp]; 
    HMVCAPoutall    = [HMVCAPoutall; HMVCAPout]; 
    HMVCAPRET       = [HMVCAPRET; price2ret(HMVCAPout,[], 'Periodic')]; 
    
    HSMVCAPout      = [1,hscappp]; 
    HSMVCAPoutall   = [HSMVCAPoutall; HSMVCAPout]; 
    HSMVCAPRET      = [HSMVCAPRET; price2ret(HSMVCAPout,[], 'Periodic')]; 
    
    HNMVCAPout      = [1,hncappp]; 
    HNMVCAPoutall   = [HNMVCAPoutall; HNMVCAPout]; 
    HNMVCAPRET       = [HNMVCAPRET; price2ret(HNMVCAPout,[], 'Periodic')]; 

end
     
 HMVCAPRETlong     = [HMVCAPRETlong, HMVCAPRET];
 HSMVCAPRETlong    = [HSMVCAPRETlong, HSMVCAPRET];
 HNMVCAPRETlong    = [HNMVCAPRETlong, HNMVCAPRET];
 % Portfolios from C-medoids clusters
for i = 1:size(indcmc_old, 1)*size(indcmc_old, 2)


tick_cmc       = Tick_old(cell2mat(indcmc_old(i)));  
indcmc         = find(ismember(Tick,tick_cmc));

tick_cmcn       = Tick_old(cell2mat(indcmcn_old(i)));  
indcmcn         = find(ismember(Tick,tick_cmcn));

tick_cmcs       = Tick_old(cell2mat(indcmcs_old(i)));  
indcmcs         = find(ismember(Tick,tick_cmcs));

%clear ccap ccapp ccappp  cncap cncapp cncappp cscap cscapp cscappp

CMVCAPout  = []; 
ccappp   = [];
ccap{1}  = 1; 
ccapp(1) = 1;

CSMVCAPout  = []; 
cscappp   = [];
cscap{1}  = 1; 
cscapp(1) = 1;

CNMVCAPout  = []; 
cncappp   = [];
cncap{1}  = 1; 
cncapp(1) = 1;


    for l = 1:size(Data,1)  
     
        XI               = Data_old(:,(cell2mat(indcmc_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kwgt             = round(kwgt.*(10^num_dig))./(10^num_dig);
        Kwgt{i}          = kwgt;
     
        ccap{l}          = sum(cell2mat(ccap(l)));    
% %portfolio appreciation
        ccap{l+1}        = sum(cell2mat(ccap(l)).*kwgt'.*(1 + Data(l,(indcmc))'));
        ccapp            = ccap{l+1};    
        ccappp           = [ccappp,ccapp];
        
        
%%%%%%%%%%%%%
        XI               = Data_old(:,(cell2mat(indcmcn_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [knwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        knwgt             = round(knwgt.*(10^num_dig))./(10^num_dig);
        Knwgt{i}          = knwgt;
     
        cncap{l}          = sum(cell2mat(cncap(l)));    
% %portfolio appreciation
        cncap{l+1}        = sum(cell2mat(cncap(l)).*knwgt'.*(1 + Data(l,(indcmcn))'));
        cncapp            = cncap{l+1};    
        cncappp           = [cncappp,cncapp];
 %%%%%%%%%%%%%%%%%%       
        XI               = Data_old(:,(cell2mat(indcmcs_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt             = round(kswgt.*(10^num_dig))./(10^num_dig);
        Kswgt{i}          = kswgt;
     
        cscap{l}          = sum(cell2mat(cscap(l)));    
% %portfolio appreciation
        cscap{l+1}        = sum(cell2mat(cscap(l)).*kswgt'.*(1 + Data(l,(indcmcs))'));
        cscapp            = cscap{l+1};    
        cscappp            = [cscappp,cscapp];
    end
    
    CMVCAPout       = [1,ccappp]; 
    CMVCAPoutall    = [CMVCAPoutall; CMVCAPout]; 
    CMVCAPRET       = [CMVCAPRET; price2ret(CMVCAPout,[], 'Periodic')]; 
    
    CSMVCAPout      = [1,cscappp]; 
    CSMVCAPoutall   = [CSMVCAPoutall; CSMVCAPout]; 
    CSMVCAPRET      = [CSMVCAPRET; price2ret(CSMVCAPout,[], 'Periodic')]; 
    
    CNMVCAPout      = [1,cncappp]; 
    CNMVCAPoutall   = [CNMVCAPoutall; CNMVCAPout]; 
    CNMVCAPRET       = [CNMVCAPRET; price2ret(CNMVCAPout,[], 'Periodic')]; 

end
     
 CMVCAPRETlong     = [CMVCAPRETlong, CMVCAPRET];
 CSMVCAPRETlong    = [CSMVCAPRETlong, CSMVCAPRET];
 CNMVCAPRETlong    = [CNMVCAPRETlong, CNMVCAPRET];
 
YEAR(n).KMVCAPout     = KMVCAPoutall;
YEAR(n).FMVCAPout     = FMVCAPoutall;
YEAR(n).HMVCAPout     = HMVCAPoutall;
YEAR(n).CMVCAPout     = CMVCAPoutall;


YEAR(n).KSMVCAPout  = KSMVCAPoutall;
YEAR(n).FSMVCAPout  = FSMVCAPoutall;
YEAR(n).HSMVCAPout  = HSMVCAPoutall;
YEAR(n).CSMVCAPout  = CSMVCAPoutall;

YEAR(n).KNMVCAPout  = KNMVCAPoutall;
YEAR(n).FNMVCAPout  = FNMVCAPoutall;
YEAR(n).HNMVCAPout  = HNMVCAPoutall;
YEAR(n).CNMVCAPout  = CNMVCAPoutall;

YEAR(n).KMVCAPRET   = KMVCAPRET;
YEAR(n).FMVCAPRET   = FMVCAPRET;
YEAR(n).HMVCAPRET   = HMVCAPRET;
YEAR(n).CMVCAPRET   = CMVCAPRET;

YEAR(n).KSMVCAPRET  = KSMVCAPRET;
YEAR(n).FSMVCAPRET  = FSMVCAPRET;
YEAR(n).HSMVCAPRET  = HSMVCAPRET;
YEAR(n).CSMVCAPRET  = CSMVCAPRET;

YEAR(n).KNMVCAPRET  = KNMVCAPRET;
YEAR(n).FNMVCAPRET  = FNMVCAPRET;
YEAR(n).HNMVCAPRET  = HNMVCAPRET;
YEAR(n).CNMVCAPRET  = CNMVCAPRET;

nshift = nshift + 1
end
toc %ca 

%% Construction of long mean-variance portfolios' cumulative returns 
tic
% k-means


    KMVCAPoutlong = [];
    KSMVCAPoutlong = [];    
    KNMVCAPoutlong = [];

parfor j = 1:size(KMVCAPRETlong,1)
    IND = [];
    IND = KMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));
 
    INDMVCAPIT    = [INDMVCAP];    
    
       end
   KMVCAPoutlong = [KMVCAPoutlong; INDMVCAPIT];
end   
parfor j = 1:size(KMVCAPRETlong,1)
    IND = [];
    IND = KNMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));

    INDMVCAPIT    = [INDMVCAP];    
    
       end
   KNMVCAPoutlong = [KNMVCAPoutlong; INDMVCAPIT];
end   
parfor j = 1:size(KMVCAPRETlong,1)
    IND = [];
    IND = KSMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));
  
    INDMVCAPIT  = [INDMVCAP];    
    
       end
   KSMVCAPoutlong = [KSMVCAPoutlong; INDMVCAPIT];
end   

% C-medoids
    CMVCAPoutlong = [];
    CSMVCAPoutlong = [];    
    CNMVCAPoutlong = [];
parfor j = 1:size(CMVCAPRETlong,1)
    IND = [];
    IND = CMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));
   % indcaplast  = INDMVCAP(i+1);
    
    INDMVCAPIT    = [INDMVCAP];    
    
       end
   CMVCAPoutlong = [CMVCAPoutlong; INDMVCAPIT];
end   
parfor j = 1:size(CMVCAPRETlong,1)
    IND = [];
    IND = CNMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));

    INDMVCAPIT    = [INDMVCAP];    
    
       end
   CNMVCAPoutlong = [CNMVCAPoutlong; INDMVCAPIT];
end   
parfor j = 1:size(CMVCAPRETlong,1)
    IND = [];
    IND = CSMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));
  
    INDMVCAPIT  = [INDMVCAP];    
    
       end
   CSMVCAPoutlong = [CSMVCAPoutlong; INDMVCAPIT];
end   

% Agglomerative clustering
    HMVCAPoutlong = [];
    HSMVCAPoutlong = [];    
    HNMVCAPoutlong = [];
parfor j = 1:size(HMVCAPRETlong,1)
    IND = [];
    IND = HMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));
   
    
    INDMVCAPIT    = [INDMVCAP];    
   
       end
   HMVCAPoutlong = [HMVCAPoutlong; INDMVCAPIT];
end   
parfor j = 1:size(HMVCAPRETlong,1)
    IND = [];
    IND = HNMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));

    INDMVCAPIT    = [INDMVCAP];    
    
       end
   HNMVCAPoutlong = [HNMVCAPoutlong; INDMVCAPIT];
end   
parfor j = 1:size(HMVCAPRETlong,1)
    IND = [];
    IND = HSMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];

    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));
  
    INDMVCAPIT  = [INDMVCAP];    
    
       end
   HSMVCAPoutlong = [HSMVCAPoutlong; INDMVCAPIT];
end    

% FUZZY clustering


    FMVCAPoutlong = [];
    FSMVCAPoutlong = [];    
    FNMVCAPoutlong = [];
parfor j = 1:size(FMVCAPRETlong,1)
    IND = [];
    IND = FMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));
   
    INDMVCAPIT    = [INDMVCAP];    
    
       end
   FMVCAPoutlong = [FMVCAPoutlong; INDMVCAPIT];
end   
parfor j = 1:size(FMVCAPRETlong,1)
    IND = [];
    IND = FNMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));

    INDMVCAPIT    = [INDMVCAP];    
    
       end
   FNMVCAPoutlong = [FNMVCAPoutlong; INDMVCAPIT];
end   
parfor j = 1:size(FMVCAPRETlong,1)
    IND = [];
    IND = FSMVCAPRETlong(j,:);
    INDMVCAP  = [];
    INDMVCAPIT  = [];
    
    
    
    for i = 2:size(IND,2)
           
    INDMVCAP(1) = 1;       
    
    INDMVCAP(i) = INDMVCAP(i-1)*(1+IND(i));
  
    INDMVCAPIT  = [INDMVCAP];    
    
       end
   FSMVCAPoutlong = [FSMVCAPoutlong; INDMVCAPIT];
end   

toc %ca 

%% Construction of long mean-variance portfolios' cumulative returns with TC
tic

nshift = 0
    KMVCAPTCoutlong  = YEAR(2).KMVCAPout(:,2:end);
    KSMVCAPTCoutlong = YEAR(2).KSMVCAPout(:,2:end);     
    KNMVCAPTCoutlong = YEAR(2).KNMVCAPout(:,2:end);
    
    HMVCAPTCoutlong  = YEAR(2).HMVCAPout(:,2:end);
    HSMVCAPTCoutlong = YEAR(2).HSMVCAPout(:,2:end);   
    HNMVCAPTCoutlong = YEAR(2).HNMVCAPout(:,2:end);
    
    CMVCAPTCoutlong  = YEAR(2).CMVCAPout(:,2:end);
    CSMVCAPTCoutlong = YEAR(2).CSMVCAPout(:,2:end);    
    CNMVCAPTCoutlong = YEAR(2).CNMVCAPout(:,2:end);
    
    FMVCAPTCoutlong  = YEAR(2).FMVCAPout(:,2:end);
    FSMVCAPTCoutlong = YEAR(2).FSMVCAPout(:,2:end);   
    FNMVCAPTCoutlong = YEAR(2).FNMVCAPout(:,2:end);
   
for n = 3:length(RET_YEAR)-1
   

    %k-means
    INDCAPTCIT    = [];
    for j = 1:size(KMVCAPRETlong,1)
           IND = [];
           IND = YEAR(n).KMVCAPRET(j,2:end); 
           INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
                
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    KMVCAPTCoutlong = [KMVCAPTCoutlong, INDCAPTCIT];
   
    INDCAPTCIT    = [];      
    for j = 1:size(KNMVCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).KNMVCAPRET(j,2:end); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    KNMVCAPTCoutlong = [KNMVCAPTCoutlong, INDCAPTCIT];  
    
    INDCAPTCIT    = []; 
    for j = 1:size(KSMVCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).KSMVCAPRET(j,2:end); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    KSMVCAPTCoutlong = [KSMVCAPTCoutlong, INDCAPTCIT];  
     
% C-medoids

    INDCAPTCIT    = [];
    for j = 1:size(CMVCAPRETlong,1)
           IND = [];
           IND = YEAR(n).CMVCAPRET(j,2:end); 
           INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
                
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    CMVCAPTCoutlong = [CMVCAPTCoutlong, INDCAPTCIT];
   
    INDCAPTCIT    = [];      
    for j = 1:size(CNMVCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).CNMVCAPRET(j,2:end); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    CNMVCAPTCoutlong = [CNMVCAPTCoutlong, INDCAPTCIT];  
    
    INDCAPTCIT    = []; 
    for j = 1:size(CSMVCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).CSMVCAPRET(j,2:end); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    CSMVCAPTCoutlong = [CSMVCAPTCoutlong, INDCAPTCIT];   
 
% Agglomerative clustering
    INDCAPTCIT    = [];
    for j = 1:size(HMVCAPRETlong,1)
           IND = [];
           IND = YEAR(n).HMVCAPRET(j,2:end); 
           INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
                
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    HMVCAPTCoutlong = [HMVCAPTCoutlong, INDCAPTCIT];
   
    INDCAPTCIT    = [];      
    for j = 1:size(HNMVCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).HNMVCAPRET(j,2:end); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    HNMVCAPTCoutlong = [HNMVCAPTCoutlong, INDCAPTCIT];  
    
    INDCAPTCIT    = []; 
    for j = 1:size(HSMVCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).HSMVCAPRET(j,2:end); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    HSMVCAPTCoutlong = [HSMVCAPTCoutlong, INDCAPTCIT];   

% FUZZY clustering
    INDCAPTCIT    = [];
    for j = 1:size(FMVCAPRETlong,1)
           IND = [];
           IND = YEAR(n).FMVCAPRET(j,2:end); 
           INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
                
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    FMVCAPTCoutlong = [FMVCAPTCoutlong, INDCAPTCIT];
   
    INDCAPTCIT    = [];      
    for j = 1:size(FNMVCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).FNMVCAPRET(j,2:end); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    FNMVCAPTCoutlong = [FNMVCAPTCoutlong, INDCAPTCIT];  
    
    INDCAPTCIT    = []; 
    for j = 1:size(FSMVCAPRETlong,1)
         
         IND = [];
         IND = YEAR(n).FSMVCAPRET(j,2:end); 
         INDCAPTC    = []; 
         
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FMVCAPTCoutlong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    FSMVCAPTCoutlong = [FSMVCAPTCoutlong, INDCAPTCIT];  
 

nshift = nshift + 1 
end 
toc

%% Buy&hold STOXX 600 NA for the period 
INDCAPITlong       = [];
IND       = INDCAPRETlong(size(YEAR(1).DATA,1)+1:end,:);
nshift = 0

INDCAP(1) = 1;
INDCAPIT  = [];

for i = 1:size(IND,1)
    
    INDCAP(i+1) = INDCAP(i)*(1+IND(i));
    indcaplast  = INDCAP(i+1);
    INDCAPIT    = [INDCAPIT,indcaplast];      
end

INDCAPITlong       = [INDCAPITlong, INDCAPIT];
%%  Portfolio construction from random stocks of every cluster
tic
num_sim = 100
nshift = 0

kmcdist = {'sqeuclidean', 'cityblock'}; % k-means disntaces 

hctdist = {'euclidean', 'seuclidean', 'cityblock', 'minkowski', 'mahalanobis'}; %  distances for
%agglomerative clustering

hctalgo = {'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'}; % agglomerative algorithms 

cmcdist = {'euclidean'; 'seuclidean'; 'sqeuclidean'; 'cityblock'; 'chebychev'; ...
           'mahalanobis'; 'minkowski'}; % c-medoids disntaces

for n=1:length(RET_YEAR)-1
    
    idx           = YEAR(n).IDX;
    idxn          = YEAR(n).IDX_N;
    idxs          = YEAR(n).IDX_S;

    fcm           = YEAR(n).FCM;
    fcmn          = YEAR(n).FCM_N;
    fcms          = YEAR(n).FCM_S;

    hct           = YEAR(n).HCT;
    hctn          = YEAR(n).HCT_N;
    hcts          = YEAR(n).HCT_S;

    cmc           = YEAR(n).CMC;
    cmcn          = YEAR(n).CMC_N;
    cmcs          = YEAR(n).CMC_S;

    RANDINDKM      = [];
    RANDINDFCM     = [];
    RANDINDHCT     = [];
    RANDINDCMC     = [];

    RANDINDKM_N    = [];
    RANDINDFCM_N   = [];
    RANDINDHCT_N   = [];
    RANDINDCMC_N   = [];

    RANDINDKM_S    = [];
    RANDINDFCM_S   = [];
    RANDINDHCT_S   = [];
    RANDINDCMC_S   = [];
    
    for l = 1:num_sim 
 
        %K-means
        RANDINDKMJ       = [];
        RANDINDKMJ_S     = [];
        RANDINDKMJ_N     = []; 
    
        for j=1:length(kmcdist)
            indkm  = {};
            indkmn = {}; 
            indkms = {};

            idx     = YEAR(n).IDX(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
            idxn    = YEAR(n).IDX_N(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
            idxs    = YEAR(n).IDX_S(:,1+(clus_num-1)*(j-1):j*(clus_num-1));

            for i = 1:size(idx, 2)
                randindkm  = [];
                randindkmn = [];
                randindkms = []; 
            
                for m = 1:size(unique(idx(:,i))',2)
                
                    K = find(idx(:,i)==m);
                    F = find(idxn(:,i)==m);
                    H = find(idxs(:,i)==m);
                
                    randindkm(m)  = K(randi(numel(K)));
                    randindkmn(m) = F(randi(numel(F)));
                    randindkms(m) = H(randi(numel(H)));

                end
                Randindkm{i}  = randindkm;
                Randindkmn{i} = randindkmn;
                Randindkms{i} = randindkms;
            end

       RANDINDKMJ       = [RANDINDKMJ; Randindkm];
       RANDINDKMJ_S     = [RANDINDKMJ_S; Randindkms];
       RANDINDKMJ_N     = [RANDINDKMJ_N; Randindkmn];   

    end

   RANDINDKM       = [RANDINDKM; RANDINDKMJ];
   RANDINDKM_S     = [RANDINDKM_S; RANDINDKMJ_S];
   RANDINDKM_N     = [RANDINDKM_N; RANDINDKMJ_N];   


      
        %FUZZY clusters
        RANDINDFCMJ        = [];
        RANDINDFCMJ_S     = [];
        RANDINDFCMJ_N     = []; 
    
       

          fcmidx  = YEAR(n).FCM;
          fcmn    = YEAR(n).FCM_N;
          fcms    = YEAR(n).FCM_S;

            for i = 1:size(idx, 2)
                randindfcm  = [];
                randindfcmn = [];
                randindfcms = []; 
            
                for m = 1:size(unique(fcmidx(:,i))',2)
                
                    K = find(idx(:,i)==m);
                    F = find(fcmn(:,i)==m);
                    H = find(fcms(:,i)==m);
                
                    randindfcm(m)  = K(randi(numel(K)));
                    randindfcmn(m) = F(randi(numel(F)));
                    randindfcms(m) = H(randi(numel(H)));

                end
                Randindfcm{i}  = randindfcm;
                Randindfcmn{i} = randindfcmn;
                Randindfcms{i} = randindfcms;
            end
       RANDINDFCM       = [RANDINDFCM; Randindfcm];
       RANDINDFCM_S     = [RANDINDFCM_S; Randindfcms];
       RANDINDFCM_N     = [RANDINDFCM_N; Randindfcmn];  

 % Agglomerative 
        RANDINDHCTJ       = [];
        RANDINDHCTJ_S     = [];
        RANDINDHCTJ_N     = []; 
    
        for j=1:size(HCT_S,1)*size(HCT_S,2)
            indhct  = {};
            indhctn = {}; 
            indhcts = {};

            hct    = cell2mat(YEAR(n).HCT(j));
            hctn   = cell2mat(YEAR(n).HCT_N(j));
            hcts   = cell2mat(YEAR(n).HCT_S(j));

            for i = 1:size(hct, 2)
                randindhct  = [];
                randindhctn = [];
                randindhcts = []; 
            
                for m = 1:size(unique(hct(:,i))',2)
                
                    K = find(hct(:,i)==m);
                    F = find(hctn(:,i)==m);
                    H = find(hcts(:,i)==m);
                
                    randindhct(m)  = K(randi(numel(K)));
                    randindhctn(m) = F(randi(numel(F)));
                    randindhcts(m) = H(randi(numel(H)));

                end
                Randindhct{i}  = randindhct;
                Randindhctn{i} = randindhctn;
                Randindhcts{i} = randindhcts;
            end

       RANDINDHCTJ       = [RANDINDHCTJ; Randindhct];
       RANDINDHCTJ_S     = [RANDINDHCTJ_S; Randindhcts];
       RANDINDHCTJ_N     = [RANDINDHCTJ_N; Randindhctn];   

        end

    RANDINDHCT       = [RANDINDHCT; RANDINDHCTJ];
    RANDINDHCT_S     = [RANDINDHCT_S; RANDINDHCTJ_S];
    RANDINDHCT_N     = [RANDINDHCT_N; RANDINDHCTJ_N];     
   
    % CMC-clusters
        RANDINDCMCJ       = [];
        RANDINDCMCJ_S     = [];
        RANDINDCMCJ_N     = []; 
    
        for j=1:length(cmcdist)
            indcmc  = {};
            indcmcn = {}; 
            indcmcs = {};
            
            cmc     = YEAR(n).CMC(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
            cmcs    = YEAR(n).CMC_N(:,1+(clus_num-1)*(j-1):j*(clus_num-1));
            cmcn    = YEAR(n).CMC_S(:,1+(clus_num-1)*(j-1):j*(clus_num-1));

            for i = 1:size(cmc, 2)
                randindcmc  = [];
                randindcmcn = [];
                randindcmcs = []; 
            
                for m = 1:size(unique(cmc(:,i))',2)
                
                    K = find(cmc(:,i)==m);
                    F = find(cmcn(:,i)==m);
                    H = find(cmcs(:,i)==m);
                
                    randindcmc(m)  = K(randi(numel(K)));
                    randindcmcn(m) = F(randi(numel(F)));
                    randindcmcs(m) = H(randi(numel(H)));

                end
                Randindcmc{i}  = randindcmc;
                Randindcmcn{i} = randindcmcn;
                Randindcmcs{i} = randindcmcs;
            end

       RANDINDCMCJ       = [RANDINDCMCJ; Randindcmc];
       RANDINDCMCJ_S     = [RANDINDCMCJ_S; Randindcmcs];
       RANDINDCMCJ_N     = [RANDINDCMCJ_N; Randindcmcn];   

    end

   RANDINDCMC       = [RANDINDCMC; RANDINDCMCJ];
   RANDINDCMC_S     = [RANDINDCMC_S; RANDINDCMCJ_S];
   RANDINDCMC_N     = [RANDINDCMC_N; RANDINDCMCJ_N];  
 end
YEAR_RAND(n).RANDINDKM     = RANDINDKM;
YEAR_RAND(n).RANDINDKM_S   = RANDINDKM_S;
YEAR_RAND(n).RANDINDKM_N   = RANDINDKM_N;

YEAR_RAND(n).RANDINDFCM     = RANDINDFCM;
YEAR_RAND(n).RANDINDFCM_S   = RANDINDFCM_S;
YEAR_RAND(n).RANDINDFCM_N   = RANDINDFCM_N;

YEAR_RAND(n).RANDINDHCT     = RANDINDHCT;
YEAR_RAND(n).RANDINDHCT_S   = RANDINDHCT_S;
YEAR_RAND(n).RANDINDHCT_N   = RANDINDHCT_N;

YEAR_RAND(n).RANDINDCMC     = RANDINDCMC;
YEAR_RAND(n).RANDINDCMC_S   = RANDINDCMC_S;
YEAR_RAND(n).RANDINDCMC_N   = RANDINDCMC_N;

nshift = nshift + 1
end
toc 

%% Random out of sample 1/n rule portfolios  from every cluster based on absolute values 
tic

nshift = 0

 RKCAPRETlong     = [];
 RKSCAPRETlong    = [];
 RKNCAPRETlong    = [];
 
 RFCAPRETlong     = [];
 RFSCAPRETlong    = [];
 RFNCAPRETlong    = [];
 
 RHCAPRETlong     = [];
 RHSCAPRETlong    = [];
 RHNCAPRETlong    = [];
 
 RCCAPRETlong     = [];
 RCSCAPRETlong    = [];
 RCNCAPRETlong    = [];

for n = 2:length(RET_YEAR)-1
       
Data          = YEAR(n).DATA;
Tick          = YEAR(n).TICK;
Tick_old      = YEAR(n-1).TICK;

for k = 1:num_sim 
    KCAPoutall  = []; 
    KNCAPoutall = []; 
    KSCAPoutall = []; 

    KCAPRET     = [];
    KNCAPRET    = [];
    KSCAPRET    = [];
    indkm_old     = YEAR_RAND(n-1).RANDINDKM(((k-1)*size(kmcdist,2)+1):k*size(kmcdist,2),:);
    indkmn_old    = YEAR_RAND(n-1).RANDINDKM_N(((k-1)*size(kmcdist,2)+1):k*size(kmcdist,2),:);
    indkms_old    = YEAR_RAND(n-1).RANDINDKM_S(((k-1)*size(kmcdist,2)+1):k*size(kmcdist,2),:);

    indfcm_old    = YEAR_RAND(n-1).RANDINDFCM(k, :);
    indfcms_old   = YEAR_RAND(n-1).RANDINDFCM_S(k, :);
    indfcmn_old   = YEAR_RAND(n-1).RANDINDFCM_N(k, :);

    indhct_old    = YEAR_RAND(n-1).RANDINDHCT(((k-1)*length(hctdist)*length(hctalgo)+1):k*length(hctdist)*length(hctalgo),:);
    indhctn_old   = YEAR_RAND(n-1).RANDINDHCT_N(((k-1)*length(hctdist)*length(hctalgo)+1):k*length(hctdist)*length(hctalgo),:);
    indhcts_old   = YEAR_RAND(n-1).RANDINDHCT_S(((k-1)*length(hctdist)*length(hctalgo)+1):k*length(hctdist)*length(hctalgo),:);

    indcmc_old    = YEAR_RAND(n-1).RANDINDCMC(((k-1)*size(cmcdist,1)+1):k*size(cmcdist,1),:);
    indcmcn_old   = YEAR_RAND(n-1).RANDINDCMC_N(((k-1)*size(cmcdist,1)+1):k*size(cmcdist,1),:);
    indcmcs_old   = YEAR_RAND(n-1).RANDINDCMC_S(((k-1)*size(cmcdist,1)+1):k*size(cmcdist,1),:);


%Portfolios from k-means clusters 
for i = 1:size(indkm_old, 1)*size(indkm_old, 2)

tick_km        = Tick_old(cell2mat(indkm_old(i)));  
indkm          = find(ismember(Tick,tick_km));

tick_kmn       = Tick_old(cell2mat(indkmn_old(i)));  
indkmn         = find(ismember(Tick,tick_kmn));

tick_kms       = Tick_old(cell2mat(indkms_old(i)));  
indkms         = find(ismember(Tick,tick_kms));

%clear kcap kcapp kcappp  

KCAPout  = []; 
kcappp   = [];
kcap{1}  = 1; 
kcapp(1) = 1;

KSCAPout  = []; 
kscappp   = [];
kscap{1}  = 1; 
kscapp(1) = 1;

KNCAPout  = []; 
kncappp   = [];
kncap{1}  = 1; 
kncapp(1) = 1;


for l = 1:size(Data,1)  
    
     kcap{l}   = (repmat(sum(cell2mat(kcap(l)))/length(indkm),1,length(indkm)));
     kcap{l+1} = sum(cell2mat(kcap(l)).*(1 + Data(l,(indkm))));
     kcapp     = kcap{l+1};
     kcappp    = [kcappp,kcapp];  
     
     kncap{l}   = (repmat(sum(cell2mat(kncap(l)))/length(indkmn),1,length(indkmn)));
     kncap{l+1} = sum(cell2mat(kncap(l)).*(1 + Data(l,(indkmn))));
     kncapp     = kncap{l+1};
     kncappp    = [kncappp,kncapp];  
     
     kscap{l}   = (repmat(sum(cell2mat(kscap(l)))/length(indkms),1,length(indkms)));
     kscap{l+1} = sum(cell2mat(kscap(l)).*(1 + Data(l,(indkms))));
     kscapp     = kscap{l+1};
     kscappp    = [kscappp,kscapp];  
end   

     KCAPout       = [1, kcappp]; 
     KCAPoutall    = [KCAPoutall; KCAPout]; 
     KCAPRET       = [KCAPRET; price2ret(KCAPout,[], 'Periodic')];
     
     KNCAPout      = [1, kncappp];
     KNCAPoutall   = [KNCAPoutall; KNCAPout]; 
     KNCAPRET      = [KNCAPRET; price2ret(KNCAPout,[], 'Periodic')];
     
     KSCAPout      = [1,kscappp]; 
     KSCAPRET      = [KSCAPRET; price2ret(KSCAPout,[], 'Periodic')];
     KSCAPoutall   = [KSCAPoutall; KSCAPout]; 

end
     
 RKCAPRETlong     = [RKCAPRETlong, KCAPRET];
 RKSCAPRETlong    = [RKSCAPRETlong, KSCAPRET];
 RKNCAPRETlong    = [RKNCAPRETlong, KNCAPRET];

 %Portfolios from FUZZY clusters with 
 
FCAPoutall  = []; 
FNCAPoutall = []; 
FSCAPoutall = []; 

FCAPRET     = [];
FNCAPRET    = [];
FSCAPRET    = [];
 for i = 1:size(indfcm_old, 1)*size(indfcm_old, 2)

%clear fcap fcapp fcappp 

FCAPout  = []; 
fcappp   = [];
fcap{1}  = 1; 
fcapp(1) = 1;

FSCAPout  = []; 
fscappp   = [];
fscap{1}  = 1; 
fscapp(1) = 1;

FNCAPout  = []; 
fncappp   = [];
fncap{1}  = 1; 
fncapp(1) = 1;

tick_fcm       = Tick_old(cell2mat(indfcm_old(i)));  
indfcm         = find(ismember(Tick,tick_fcm));

tick_fcmn      = Tick_old(cell2mat(indfcmn_old(i)));  
indfcmn        = find(ismember(Tick,tick_fcmn));

tick_fcms      = Tick_old(cell2mat(indfcms_old(i)));  
indfcms        = find(ismember(Tick,tick_fcms));


for l = 1:size(Data,1)  
    
     fcap{l}   = (repmat(sum(cell2mat(fcap(l)))/length(indfcm),1,length(indfcm)));
     fcap{l+1} = sum(cell2mat(fcap(l)).*(1 + Data(l,(indfcm))));
     fcapp     = fcap{l+1};
     fcappp    = [fcappp,fcapp];  
     
     fncap{l}   = (repmat(sum(cell2mat(fncap(l)))/length(indfcmn),1,length(indfcmn)));
     fncap{l+1} = sum(cell2mat(fncap(l)).*(1 + Data(l,(indfcmn))));
     fncapp     = fncap{l+1};
     fncappp    = [fncappp,fncapp];  
     
     fscap{l}   = (repmat(sum(cell2mat(fscap(l)))/length(indfcms),1,length(indfcms)));
     fscap{l+1} = sum(cell2mat(fscap(l)).*(1 + Data(l,(indfcms))));
     fscapp     = fscap{l+1};
     fscappp    = [fscappp,fscapp];  
end   

     FCAPout    = [1, fcappp]; 
     FCAPoutall = [FCAPoutall; FCAPout]; 
     FCAPRET    = [FCAPRET; price2ret(FCAPout,[], 'Periodic')];
     
     FNCAPout    = [1, fncappp];
     FNCAPoutall = [FNCAPoutall; FNCAPout]; 
     FNCAPRET    = [FNCAPRET; price2ret(FNCAPout,[], 'Periodic')];
     

     FSCAPout    = [1, fscappp]; 
     FSCAPRET    = [FSCAPRET; price2ret(FSCAPout,[], 'Periodic')];
     FSCAPoutall = [FSCAPoutall; FSCAPout]; 

end
     
 RFCAPRETlong     = [RFCAPRETlong, FCAPRET];
 RFSCAPRETlong    = [RFSCAPRETlong, FSCAPRET];
 RFNCAPRETlong    = [RFNCAPRETlong, FNCAPRET];
    
%Portfolios from hct clusters with 

HCAPoutall  = []; 
HNCAPoutall = []; 
HSCAPoutall = []; 

HCAPRET     = [];
HNCAPRET    = [];
HSCAPRET    = [];
     
 for i = 1:size(indhct_old, 1)*size(indhct_old, 2)

tick_hct        = Tick_old(cell2mat(indhct_old(i)));  
indhct          = find(ismember(Tick,tick_hct));

tick_hctn       = Tick_old(cell2mat(indhctn_old(i)));  
indhctn         = find(ismember(Tick,tick_hctn));

tick_hcts       = Tick_old(cell2mat(indhcts_old(i)));  
indhcts         = find(ismember(Tick,tick_hcts));

%clear hcap hcapp hcappp 

HCAPout  = []; 
hcappp   = [];
hcap{1}  = 1; 
hcapp(1) = 1;

HSCAPout  = []; 
hscappp   = [];
hscap{1}  = 1; 
hscapp(1) = 1;

HNCAPout  = []; 
hncappp   = [];
hncap{1}  = 1; 
hncapp(1) = 1;

for l = 1:size(Data,1)  
    
     hcap{l}   = (repmat(sum(cell2mat(hcap(l)))/length(indhct),1,length(indhct)));
     hcap{l+1} = sum(cell2mat(hcap(l)).*(1 + Data(l,(indhct))));
     hcapp     = hcap{l+1};
     hcappp    = [hcappp,hcapp];  
     
     hncap{l}   = (repmat(sum(cell2mat(hncap(l)))/length(indhctn),1,length(indhctn)));
     hncap{l+1} = sum(cell2mat(hncap(l)).*(1 + Data(l,(indhctn))));
     hncapp     = hncap{l+1};
     hncappp    = [hncappp,hncapp];  
     
     hscap{l}   = (repmat(sum(cell2mat(hscap(l)))/length(indhcts),1,length(indhcts)));
     hscap{l+1} = sum(cell2mat(hscap(l)).*(1 + Data(l,(indhcts))));
     hscapp     = hscap{l+1};
     hscappp    = [hscappp,hscapp];  
end   

     HCAPout    = [1, hcappp]; 
     HCAPoutall = [HCAPoutall; HCAPout]; 
     HCAPRET    = [HCAPRET; price2ret(HCAPout,[], 'Periodic')];
     
     HNCAPout    = [1, hncappp];
     HNCAPoutall = [HNCAPoutall; HNCAPout]; 
     HNCAPRET    = [HNCAPRET; price2ret(HNCAPout,[], 'Periodic')];

     HSCAPout    = [1, hscappp]; 
     HSCAPRET    = [HSCAPRET; price2ret(HSCAPout,[], 'Periodic')];
     HSCAPoutall = [HSCAPoutall; HSCAPout]; 

end
     
 RHCAPRETlong     = [RHCAPRETlong, HCAPRET];
 RHSCAPRETlong    = [RHSCAPRETlong, HSCAPRET];
 RHNCAPRETlong    = [RHNCAPRETlong, HNCAPRET];
 
 % Portfolios from C-medoids clusters
 
CCAPoutall  = []; 
CNCAPoutall = []; 
CSCAPoutall = []; 

CCAPRET     = [];
CNCAPRET    = [];
CSCAPRET    = [];

for i = 1:size(indcmc_old, 1)*size(indcmc_old, 2)
    
tick_cmc       = Tick_old(cell2mat(indcmc_old(i)));  
indcmc         = find(ismember(Tick,tick_cmc));

tick_cmcn       = Tick_old(cell2mat(indcmcn_old(i)));  
indcmcn         = find(ismember(Tick,tick_cmcn));

tick_cmcs       = Tick_old(cell2mat(indcmcs_old(i)));  
indcmcs         = find(ismember(Tick,tick_cmcs));

%clear ccap ccapp ccappp  

CCAPout  = []; 
ccappp   = [];
ccap{1}  = 1; 
ccapp(1) = 1;

CSCAPout  = []; 
cscappp   = [];
cscap{1}  = 1; 
cscapp(1) = 1;

CNCAPout  = []; 
cncappp   = [];
cncap{1}  = 1; 
cncapp(1) = 1;


     for l = 1:size(Data,1)  
    
     ccap{l}   = (repmat(sum(cell2mat(ccap(l)))/length(indcmc),1,length(indcmc)));
     ccap{l+1} = sum(cell2mat(ccap(l)).*(1 + Data(l,(indcmc))));
     ccapp     = ccap{l+1};
     ccappp    = [ccappp,ccapp];  
     
     cncap{l}   = (repmat(sum(cell2mat(cncap(l)))/length(indcmcn),1,length(indcmcn)));
     cncap{l+1} = sum(cell2mat(cncap(l)).*(1 + Data(l,(indcmcn))));
     cncapp     = cncap{l+1};
     cncappp    = [cncappp,cncapp];  
     
     cscap{l}   = (repmat(sum(cell2mat(cscap(l)))/length(indcmcs),1,length(indcmcs)));
     cscap{l+1} = sum(cell2mat(cscap(l)).*(1 + Data(l,(indcmcs))));
     cscapp     = cscap{l+1};
     cscappp    = [cscappp,cscapp];  
     end   

     CCAPout    = [1,ccappp]; 
     CCAPoutall = [CCAPoutall; CCAPout]; 
     CCAPRET    = [CCAPRET; price2ret(CCAPout,[], 'Periodic')];
     
     CNCAPout    = [1,cncappp];
     CNCAPoutall   = [CNCAPoutall; CNCAPout]; 
     CNCAPRET    = [CNCAPRET; price2ret(CNCAPout,[], 'Periodic')];
     
     CSCAPout    = [1,cscappp]; 
     CSCAPRET    = [CSCAPRET; price2ret(CSCAPout,[], 'Periodic')];
     CSCAPoutall = [CSCAPoutall; CSCAPout]; 

end
     
 RCCAPRETlong     = [RCCAPRETlong, CCAPRET];
 RCSCAPRETlong    = [RCSCAPRETlong, CSCAPRET];
 RCNCAPRETlong    = [RCNCAPRETlong, CNCAPRET];

RKCAPout{k}    = KCAPoutall;
RFCAPout{k}    = FCAPoutall;
RHCAPout{k}    = HCAPoutall;
RCCAPout{k}    = CCAPoutall;

RKSCAPout{k} = KSCAPoutall;
RFSCAPout{k} = FSCAPoutall;
RHSCAPout{k} = HSCAPoutall;
RCSCAPout{k} = CSCAPoutall;

RKNCAPout{k} = KNCAPoutall;
RFNCAPout{k} = FNCAPoutall;
RHNCAPout{k} = HNCAPoutall;
RCNCAPout{k} = CNCAPoutall;

RKCAPRET{k}  = KCAPRET;
RFCAPRET{k}  = FCAPRET;
RHCAPRET{k}  = HCAPRET;
RCCAPRET{k}  = CCAPRET;

RKSCAPRET{k} = KSCAPRET;
RFSCAPRET{k} = FSCAPRET;
RHSCAPRET{k} = HSCAPRET;
RCSCAPRET{k} = CSCAPRET;

RKNCAPRET{k} = KNCAPRET;
RFNCAPRET{k} = FNCAPRET;
RHNCAPRET{k} = HNCAPRET;
RCNCAPRET{k} = CNCAPRET;

end
YEAR_RAND(n).RKCAPout   = RKCAPout;
YEAR_RAND(n).RFCAPout   = RFCAPout;
YEAR_RAND(n).RHCAPout   = RHCAPout;
YEAR_RAND(n).RCCAPout   = RCCAPout;

YEAR_RAND(n).RKSCAPout  = RKSCAPout;
YEAR_RAND(n).RFSCAPout  = RFSCAPout;
YEAR_RAND(n).RHSCAPout  = RHSCAPout;
YEAR_RAND(n).RCSCAPout  = RCSCAPout;

YEAR_RAND(n).RKNCAPout  = RKNCAPout;
YEAR_RAND(n).RFNCAPout  = RFNCAPout;
YEAR_RAND(n).RHNCAPout  = RHNCAPout;
YEAR_RAND(n).RCNCAPout  = RCNCAPout;

YEAR_RAND(n).RKCAPRET   = RKCAPRET;
YEAR_RAND(n).RFCAPRET   = RFCAPRET;
YEAR_RAND(n).RHCAPRET   = RHCAPRET;
YEAR_RAND(n).RCCAPRET   = RCCAPRET;

YEAR_RAND(n).RKSCAPRET  = RKSCAPRET;
YEAR_RAND(n).RFSCAPRET  = RFSCAPRET;
YEAR_RAND(n).RHSCAPRET  = RHSCAPRET;
YEAR_RAND(n).RCSCAPRET  = RCSCAPRET;

YEAR_RAND(n).RKNCAPRET  = RKNCAPRET;
YEAR_RAND(n).RFNCAPRET  = RFNCAPRET;
YEAR_RAND(n).RHNCAPRET  = RHNCAPRET;
YEAR_RAND(n).RCNCAPRET  = RCNCAPRET;

nshift = nshift + 1
end
toc %ca 
%% Random out of sample MV rule portfolios  from every cluster based on absolute values 
tic

nshift = 0

 RKMVCAPRETlong     = [];
 RKSMVCAPRETlong    = [];
 RKNMVCAPRETlong    = [];
 
 RFMVCAPRETlong     = [];
 RFSMVCAPRETlong    = [];
 RFNMVCAPRETlong    = [];
 
 RHMVCAPRETlong     = [];
 RHSMVCAPRETlong    = [];
 RHNMVCAPRETlong    = [];
 
 RCMVCAPRETlong     = [];
 RCSMVCAPRETlong    = [];
 RCNMVCAPRETlong    = [];

for n = 2:length(RET_YEAR)-1
       
Data          = YEAR(n).DATA;
Data_old      = YEAR(n-1).DATA;
Tick          = YEAR(n).TICK;
Tick_old      = YEAR(n-1).TICK;

indkm_old     = YEAR(n-1).INDKM;
indkmn_old    = YEAR(n-1).INDKM_N;
indkms_old    = YEAR(n-1).INDKM_S;

indfcm_old    = YEAR(n-1).INDFCM;
indfcms_old   = YEAR(n-1).INDFCM_S;
indfcmn_old   = YEAR(n-1).INDFCM_N;

indhct_old    = YEAR(n-1).INDHCT;
indhctn_old   = YEAR(n-1).INDHCT_N;
indhcts_old   = YEAR(n-1).INDHCT_S;

indcmc_old    = YEAR(n-1).INDCMC;
indcmcn_old   = YEAR(n-1).INDCMC_N;
indcmcs_old   = YEAR(n-1).INDCMC_S;

num_dig       = 4;
TargRet       = 0.8;
options       = optimset('Algorithm','active-set','MaxFunEvals',100000);

KWGT  = {}; %weights for MV allocation
KSWGT = {};
KNWGT = {};

FWGT  = {}; %weights for MV allocation
FSWGT = {};
FNWGT = {};

HWGT  = {}; %weights for MV allocation
HSWGT = {};
HNWGT = {};

CWGT  = {}; %weights for MV allocation
CSWGT = {};
CNWGT = {};

for k     = 1:num_sim
    RKCAPout  = [];
    RKNCAPout = [];
    RKSCAPout = [];

    RFCAPout  = [];
    RFNCAPout = [];
    RFSCAPout = [];

    RCCAPout  = [];
    RCNCAPout = [];
    RCSCAPout = [];

    RHCAPout  = [];
    RHNCAPout = [];
    RHSCAPout = [];

    KCAPoutall  = []; 
    KNCAPoutall = []; 
    KSCAPoutall = []; 

    RKCAPRET  = [];
    RKSCAPRET = [];
    RKNCAPRET = [];

    RFCAPRET  = [];
    RFSCAPRET = [];
    RFNCAPRET = [];

    RCCAPRET  = [];
    RCSCAPRET = [];
    RCNCAPRET = [];

    RHCAPRET  = [];
    RHSCAPRET = [];
    RHNCAPRET = [];

    KCAPRET   = [];
    KNCAPRET  = [];
    KSCAPRET  = [];

    indkm_old     = YEAR_RAND(n-1).RANDINDKM(((k-1)*size(kmcdist,2)+1):k*size(kmcdist,2),:);
    indkmn_old    = YEAR_RAND(n-1).RANDINDKM_N(((k-1)*size(kmcdist,2)+1):k*size(kmcdist,2),:);
    indkms_old    = YEAR_RAND(n-1).RANDINDKM_S(((k-1)*size(kmcdist,2)+1):k*size(kmcdist,2),:);

    indfcm_old    = YEAR_RAND(n-1).RANDINDFCM(k, :);
    indfcms_old   = YEAR_RAND(n-1).RANDINDFCM_S(k, :);
    indfcmn_old   = YEAR_RAND(n-1).RANDINDFCM_N(k, :);

    indhct_old    = YEAR_RAND(n-1).RANDINDHCT(((k-1)*length(hctdist)*length(hctalgo)+1):k*length(hctdist)*length(hctalgo),:);
    indhctn_old   = YEAR_RAND(n-1).RANDINDHCT_N(((k-1)*length(hctdist)*length(hctalgo)+1):k*length(hctdist)*length(hctalgo),:);
    indhcts_old   = YEAR_RAND(n-1).RANDINDHCT_S(((k-1)*length(hctdist)*length(hctalgo)+1):k*length(hctdist)*length(hctalgo),:);

    indcmc_old    = YEAR_RAND(n-1).RANDINDCMC(((k-1)*size(cmcdist,1)+1):k*size(cmcdist,1),:);
    indcmcn_old   = YEAR_RAND(n-1).RANDINDCMC_N(((k-1)*size(cmcdist,1)+1):k*size(cmcdist,1),:);
    indcmcs_old   = YEAR_RAND(n-1).RANDINDCMC_S(((k-1)*size(cmcdist,1)+1):k*size(cmcdist,1),:);


%Portfolios from k-means clusters 
for i = 1:size(indkm_old, 1)*size(indkm_old, 2)

tick_km        = Tick_old(cell2mat(indkm_old(i)));  
indkm          = find(ismember(Tick,tick_km));

tick_kmn       = Tick_old(cell2mat(indkmn_old(i)));  
indkmn         = find(ismember(Tick,tick_kmn));

tick_kms       = Tick_old(cell2mat(indkms_old(i)));  
indkms         = find(ismember(Tick,tick_kms));

clear kcap kcapp kcappp  

KCAPout  = []; 
kcappp   = [];
kcap{1}  = 1; 
kcapp(1) = 1;

KSCAPout  = []; 
kscappp   = [];
kscap{1}  = 1; 
kscapp(1) = 1;

KNCAPout  = []; 
kncappp   = [];
kncap{1}  = 1; 
kncapp(1) = 1;


for l = 1:size(Data,1)  
       XI               = Data_old(:,(cell2mat(indkm_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kwgt             = round(kwgt.*(10^num_dig))./(10^num_dig);
        Kwgt{i}          = kwgt;
        kcap{l}          = sum(cell2mat(kcap(l)));    
%portfolio appreciation
        kcap{l+1}        = sum(cell2mat(kcap(l)).*kwgt'.*(1 + Data(l,(indkm))'));
        kcapp            = kcap{l+1};    
        kcappp           = [kcappp,kcapp];
        
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        XI               = Data_old(:,(cell2mat(indkmn_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [knwgt,iVaR]     = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        knwgt            = round(knwgt.*(10^num_dig))./(10^num_dig);
        Knwgt{i}         = knwgt;
     
        kncap{l}         = sum(cell2mat(kncap(l)));    
% %portfolio appreciation
        kncap{l+1}       = sum(cell2mat(kncap(l)).*knwgt'.*(1 + Data(l,(indkmn))'));
        kncapp           = kncap{l+1};    
        kncappp          = [kncappp,kncapp];
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        XI               = Data_old(:,(cell2mat(indkms_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]     = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt            = round(kswgt.*(10^num_dig))./(10^num_dig);
        Kswgt{i}         = kswgt;
     
        kscap{l}         = sum(cell2mat(kscap(l)));    
%portfolio appreciation
        kscap{l+1}       = sum(cell2mat(kscap(l)).*kswgt'.*(1 + Data(l,(indkms))'));
        kscapp           = kscap{l+1};    
        kscappp          = [kscappp,kscapp];
end   

     KCAPout       = [1, kcappp]; 
     KCAPoutall    = [KCAPoutall; KCAPout]; 
     KCAPRET       = [KCAPRET; price2ret(KCAPout,[], 'Periodic')];
     
     KNCAPout      = [1, kncappp];
     KNCAPoutall   = [KNCAPoutall; KNCAPout]; 
     KNCAPRET      = [KNCAPRET; price2ret(KNCAPout,[], 'Periodic')];
     
     KSCAPout      = [1,kscappp]; 
     KSCAPRET      = [KSCAPRET; price2ret(KSCAPout,[], 'Periodic')];
     KSCAPoutall   = [KSCAPoutall; KSCAPout]; 

end
     
 RKMVCAPRETlong     = [RKMVCAPRETlong, KCAPRET];
 RKSMVCAPRETlong    = [RKMVCAPRETlong, KSCAPRET];
 RKNMVCAPRETlong    = [RKNMVCAPRETlong, KNCAPRET];

% Portfolios from FUZZY clusters with 
 
FCAPoutall  = []; 
FNCAPoutall = []; 
FSCAPoutall = []; 

FCAPRET     = [];
FNCAPRET    = [];
FSCAPRET    = [];
   for i = 1:size(indfcm_old, 1)*size(indfcm_old, 2)

clear fcap fcapp fcappp 

FCAPout  = []; 
fcappp   = [];
fcap{1}  = 1; 
fcapp(1) = 1;

FSCAPout  = []; 
fscappp   = [];
fscap{1}  = 1; 
fscapp(1) = 1;

FNCAPout  = []; 
fncappp   = [];
fncap{1}  = 1; 
fncapp(1) = 1;

tick_fcm       = Tick_old(cell2mat(indfcm_old(i)));  
indfcm         = find(ismember(Tick,tick_fcm));

tick_fcmn      = Tick_old(cell2mat(indfcmn_old(i)));  
indfcmn        = find(ismember(Tick,tick_fcmn));

tick_fcms      = Tick_old(cell2mat(indfcms_old(i)));  
indfcms        = find(ismember(Tick,tick_fcms));


for l = 1:size(Data,1)  
    
        XI               = Data_old(:,(cell2mat(indfcm_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kwgt             = round(kwgt.*(10^num_dig))./(10^num_dig);
        Fwgt{i}          = kwgt;
     
        fcap{l}          = sum(cell2mat(fcap(l)));    
%portfolio appreciation
        fcap{l+1}        = sum(cell2mat(fcap(l)).*kwgt'.*(1 + Data(l,(indfcm))'));
        fcapp            = fcap{l+1};    
        fcappp           = [fcappp,fcapp];
        
        
%%%%%%%%%%%%
        XI               = Data_old(:,(cell2mat(indfcmn_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [knwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        knwgt             = round(knwgt.*(10^num_dig))./(10^num_dig);
        Fnwgt{i}          = knwgt;
     
        fncap{l}          = sum(cell2mat(fncap(l)));    
% %portfolio appreciation
        fncap{l+1}        = sum(cell2mat(fncap(l)).*knwgt'.*(1 + Data(l,(indfcmn))'));
        fncapp            = fncap{l+1};    
        fncappp           = [fncappp,fncapp];
 %%%%%%%%%%%%%%%%%       
        XI               = Data_old(:,(cell2mat(indfcms_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt             = round(kswgt.*(10^num_dig))./(10^num_dig);
        Fswgt{i}          = kswgt;
     
        fscap{l}          = sum(cell2mat(fscap(l)));    
%portfolio appreciation
        fscap{l+1}        = sum(cell2mat(fscap(l)).*kswgt'.*(1 + Data(l,(indfcms))'));
        fscapp            = fscap{l+1};    
        fscappp           = [fscappp,fscapp]; 
end   

     FCAPout    = [1, fcappp]; 
     FCAPoutall = [FCAPoutall; FCAPout]; 
     FCAPRET    = [FCAPRET; price2ret(FCAPout,[], 'Periodic')];
     
     FNCAPout    = [1, fncappp];
     FNCAPoutall = [FNCAPoutall; FNCAPout]; 
     FNCAPRET    = [FNCAPRET; price2ret(FNCAPout,[], 'Periodic')];
     

     FSCAPout    = [1, fscappp]; 
     FSCAPRET    = [FSCAPRET; price2ret(FSCAPout,[], 'Periodic')];
     FSCAPoutall = [FSCAPoutall; FSCAPout]; 

end
     
 RFMVCAPRETlong     = [RFMVCAPRETlong, FCAPRET];
 RFSMVCAPRETlong    = [RFSMVCAPRETlong, FSCAPRET];
 RFNMVCAPRETlong    = [RFNMVCAPRETlong, FNCAPRET];
    
% Portfolios from hct clusters with 

HCAPoutall  = []; 
HNCAPoutall = []; 
HSCAPoutall = []; 

HCAPRET     = [];
HNCAPRET    = [];
HSCAPRET    = [];
     
 for i = 1:size(indhct_old, 1)*size(indhct_old, 2)

tick_hct        = Tick_old(cell2mat(indhct_old(i)));  
indhct          = find(ismember(Tick,tick_hct));

tick_hctn       = Tick_old(cell2mat(indhctn_old(i)));  
indhctn         = find(ismember(Tick,tick_hctn));

tick_hcts       = Tick_old(cell2mat(indhcts_old(i)));  
indhcts         = find(ismember(Tick,tick_hcts));

clear hcap hcapp hcappp 

HCAPout  = []; 
hcappp   = [];
hcap{1}  = 1; 
hcapp(1) = 1;

HSCAPout  = []; 
hscappp   = [];
hscap{1}  = 1; 
hscapp(1) = 1;

HNCAPout  = []; 
hncappp   = [];
hncap{1}  = 1; 
hncapp(1) = 1;

for l = 1:size(Data,1)  
    
        XI               = Data_old(:,(cell2mat(indhct_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kwgt             = round(kwgt.*(10^num_dig))./(10^num_dig);
        Hwgt{i}          = kwgt;
     
        hcap{l}          = sum(cell2mat(hcap(l)));    
%portfolio appreciation
        hcap{l+1}        = sum(cell2mat(hcap(l)).*kwgt'.*(1 + Data(l,(indhct))'));
        hcapp            = hcap{l+1};    
        hcappp           = [hcappp,hcapp];
        
        
%%%%%%%%%%%
        XI               = Data_old(:,(cell2mat(indhctn_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [knwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        knwgt             = round(knwgt.*(10^num_dig))./(10^num_dig);
        Hnwgt{i}          = knwgt;
     
        hncap{l}          = sum(cell2mat(hncap(l)));    
%portfolio appreciation
        hncap{l+1}        = sum(cell2mat(hncap(l)).*knwgt'.*(1 + Data(l,(indhctn))'));
        hncapp            = hncap{l+1};    
        hncappp           = [hncappp,hncapp];
 %%%%%%%%%%%%%%%%       
        XI               = Data_old(:,(cell2mat(indhcts_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt             = round(kswgt.*(10^num_dig))./(10^num_dig);
        Hswgt{i}          = kswgt;
     
        hscap{l}          = sum(cell2mat(hscap(l)));    
%portfolio appreciation
        hscap{l+1}        = sum(cell2mat(hscap(l)).*kswgt'.*(1 + Data(l,(indhcts))'));
        hscapp            = hscap{l+1};    
        hscappp            = [hscappp,hscapp];
end   

     HCAPout    = [1, hcappp]; 
     HCAPoutall = [HCAPoutall; HCAPout]; 
     HCAPRET    = [HCAPRET; price2ret(HCAPout,[], 'Periodic')];
     
     HNCAPout    = [1, hncappp];
     HNCAPoutall = [HNCAPoutall; HNCAPout]; 
     HNCAPRET    = [HNCAPRET; price2ret(HNCAPout,[], 'Periodic')];

     HSCAPout    = [1, hscappp]; 
     HSCAPRET    = [HSCAPRET; price2ret(HSCAPout,[], 'Periodic')];
     HSCAPoutall = [HSCAPoutall; HSCAPout]; 

end
     
 RHMVCAPRETlong     = [RHMVCAPRETlong, HCAPRET];
 RHSMVCAPRETlong    = [RHSMVCAPRETlong, HSCAPRET];
 RHNMVCAPRETlong    = [RHNMVCAPRETlong, HNCAPRET];
 
 Portfolios from C-medoids clusters
 
CCAPoutall  = []; 
CNCAPoutall = []; 
CSCAPoutall = []; 

CCAPRET     = [];
CNCAPRET    = [];
CSCAPRET    = [];

for i = 1:size(indcmc_old, 1)*size(indcmc_old, 2)
    
tick_cmc       = Tick_old(cell2mat(indcmc_old(i)));  
indcmc         = find(ismember(Tick,tick_cmc));

tick_cmcn       = Tick_old(cell2mat(indcmcn_old(i)));  
indcmcn         = find(ismember(Tick,tick_cmcn));

tick_cmcs       = Tick_old(cell2mat(indcmcs_old(i)));  
indcmcs         = find(ismember(Tick,tick_cmcs));

clear ccap ccapp ccappp  

CCAPout  = []; 
ccappp   = [];
ccap{1}  = 1; 
ccapp(1) = 1;

CSCAPout  = []; 
cscappp   = [];
cscap{1}  = 1; 
cscapp(1) = 1;

CNCAPout  = []; 
cncappp   = [];
cncap{1}  = 1; 
cncapp(1) = 1;


     for l = 1:size(Data,1)  
    
        XI               = Data_old(:,(cell2mat(indcmc_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kwgt             = round(kwgt.*(10^num_dig))./(10^num_dig);
        Cwgt{i}          = kwgt;
     
        ccap{l}          = sum(cell2mat(ccap(l)));    
%portfolio appreciation
        ccap{l+1}        = sum(cell2mat(ccap(l)).*kwgt'.*(1 + Data(l,(indcmc))'));
        ccapp            = ccap{l+1};    
        ccappp           = [ccappp,ccapp];
        
        
%%%%%%%%%%%%
        XI               = Data_old(:,(cell2mat(indcmcn_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [knwgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        knwgt             = round(knwgt.*(10^num_dig))./(10^num_dig);
        Cnwgt{i}          = knwgt;
     
        cncap{l}          = sum(cell2mat(cncap(l)));    
% %portfolio appreciation
        cncap{l+1}        = sum(cell2mat(cncap(l)).*knwgt'.*(1 + Data(l,(indcmcn))'));
        cncapp            = cncap{l+1};    
        cncappp           = [cncappp,cncapp];
 %%%%%%%%%%%%%%%%%%       
        XI               = Data_old(:,(cell2mat(indcmcs_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt             = round(kswgt.*(10^num_dig))./(10^num_dig);
        Cswgt{i}          = kswgt;
     
        cscap{l}          = sum(cell2mat(cscap(l)));    
%portfolio appreciation
        cscap{l+1}        = sum(cell2mat(cscap(l)).*kswgt'.*(1 + Data(l,(indcmcs))'));
        cscapp            = cscap{l+1};    
        cscappp           = [cscappp,cscapp];
     end   

     CCAPout    = [1,ccappp]; 
     CCAPoutall = [CCAPoutall; CCAPout]; 
     CCAPRET    = [CCAPRET; price2ret(CCAPout,[], 'Periodic')];
     
     CNCAPout    = [1,cncappp];
     CNCAPoutall   = [CNCAPoutall; CNCAPout]; 
     CNCAPRET    = [CNCAPRET; price2ret(CNCAPout,[], 'Periodic')];
     
     CSCAPout    = [1,cscappp]; 
     CSCAPRET    = [CSCAPRET; price2ret(CSCAPout,[], 'Periodic')];
     CSCAPoutall = [CSCAPoutall; CSCAPout]; 

end
     
RCMVCAPRETlong     = [RCMVCAPRETlong, CCAPRET];
RCSMVCAPRETlong    = [RCSMVCAPRETlong, CSCAPRET];
RCNMVCAPRETlong    = [RCNMVCAPRETlong, CNCAPRET];

RKCAPout{k}    = KCAPoutall;
RFCAPout{k}    = FCAPoutall;
RHCAPout{k}    = HCAPoutall;
RCCAPout{k}    = CCAPoutall;

RKSCAPout{k} = KSCAPoutall;
RFSCAPout{k} = FSCAPoutall;
RHSCAPout{k} = HSCAPoutall;
RCSCAPout{k} = CSCAPoutall;

RKNCAPout{k} = KNCAPoutall;
RFNCAPout{k} = FNCAPoutall;
RHNCAPout{k} = HNCAPoutall;
RCNCAPout{k} = CNCAPoutall;

RKCAPRET{k}  = KCAPRET;
RFCAPRET{k}  = FCAPRET;
RHCAPRET{k}  = HCAPRET;
RCCAPRET{k}  = CCAPRET;

RKSCAPRET{k} = KSCAPRET;
RFSCAPRET{k} = FSCAPRET;
RHSCAPRET{k} = HSCAPRET;
RCSCAPRET{k} = CSCAPRET;

RKNCAPRET{k} = KNCAPRET;
RFNCAPRET{k} = FNCAPRET;
RHNCAPRET{k} = HNCAPRET;
RCNCAPRET{k} = CNCAPRET;

KWGT{k}   = Kwgt; %weights for MV allocation
KSWGT{k}  = Kswgt;
KNWGT{k}  = Knwgt;

FWGT{k}   = Fwgt; %weights for MV allocation
FSWGT{k}  = Fswgt;
FNWGT{k}  = Fnwgt;

HWGT{k}   = Hwgt; %weights for MV allocation
HSWGT{k}  = Hswgt;
HNWGT{k}  = Hnwgt;

CWGT{k}   = Cwgt; %weights for MV allocation
CSWGT{k}  = Cswgt;
CNWGT{k}  = Cnwgt;

end
YEAR_RAND(n).RKMVCAPout   = RKCAPout;
YEAR_RAND(n).RFMVCAPout   = RFCAPout;
YEAR_RAND(n).RHMVCAPout   = RHCAPout;
YEAR_RAND(n).RCMVCAPout   = RCCAPout;

YEAR_RAND(n).RKMVSCAPout  = RKSCAPout;
YEAR_RAND(n).RFMVSCAPout  = RFSCAPout;
YEAR_RAND(n).RHMVSCAPout  = RHSCAPout;
YEAR_RAND(n).RCMVSCAPout  = RCSCAPout;

YEAR_RAND(n).RKMVNCAPout  = RKNCAPout;
YEAR_RAND(n).RFMVNCAPout  = RFNCAPout;
YEAR_RAND(n).RHMVNCAPout  = RHNCAPout;
YEAR_RAND(n).RCMVNCAPout  = RCNCAPout;

YEAR_RAND(n).RKMVCAPRET   = RKCAPRET;
YEAR_RAND(n).RFMVCAPRET   = RFCAPRET;
YEAR_RAND(n).RHMVCAPRET   = RHCAPRET;
YEAR_RAND(n).RCMVCAPRET   = RCCAPRET;

YEAR_RAND(n).RKSMVCAPRET  = RKSCAPRET;
YEAR_RAND(n).RFSMVCAPRET  = RFSCAPRET;
YEAR_RAND(n).RHMVSCAPRET  = RHSCAPRET;
YEAR_RAND(n).RCMVSCAPRET  = RCSCAPRET;

YEAR_RAND(n).RKMVNCAPRET  = RKNCAPRET;
YEAR_RAND(n).RFMVNCAPRET  = RFNCAPRET;
YEAR_RAND(n).RHNMVCAPRET  = RHNCAPRET;
YEAR_RAND(n).RCNMVCAPRET  = RCNCAPRET;

YEAR_RAND(n).KWGT  = KWGT; %weights for MV allocation
YEAR_RAND(n).KSWGT = KSWGT;
YEAR_RAND(n).KNWGT = KNWGT; 

YEAR_RAND(n).FWGT  = FWGT; %weights for MV allocation
YEAR_RAND(n).FSWGT = FSWGT; 
YEAR_RAND(n).FNWGT = FNWGT;

YEAR_RAND(n).CWGT  = CWGT; %weights for MV allocation
YEAR_RAND(n).CSWGT = CSWGT; 
YEAR_RAND(n).CNWGT = CNWGT; 

YEAR_RAND(n).HWGT  = HWGT; %weights for MV allocation
YEAR_RAND(n).HSWGT = HSWGT; 
YEAR_RAND(n).HNWGT = HNWGT; 

nshift = nshift + 1
end
toc %ca 
%% Time-varying number of clusters construction of returns of max Sharpe portfolios: Silhouette criterion
%based on absolute values for the whole period effeective number of clustering due to Silhouette criterion

KCAPTCoutlong_tv  = [];
KSCAPTCoutlong_tv = [];
KNCAPTCoutlong_tv = [];

FCAPTCoutlong_tv  = [];
FSCAPTCoutlong_tv = [];
FNCAPTCoutlong_tv = [];

CCAPTCoutlong_tv  = [];
CSCAPTCoutlong_tv = [];
CNCAPTCoutlong_tv = [];

HCAPTCoutlong_tv  = [];
HSCAPTCoutlong_tv = [];
HNCAPTCoutlong_tv = [];

KMC_NUM_SIL = [];
KMCN_NUM_SIL = [];
KMCS_NUM_SIL = [];

FCM_NUM_SIL = [];
FCMN_NUM_SIL = [];
FCMS_NUM_SIL = [];

CMC_NUM_SIL = [];
CMCN_NUM_SIL = [];
CMCS_NUM_SIL = [];

HCT_NUM_SIL = [];
HCTN_NUM_SIL = [];
HCTS_NUM_SIL = [];


for i = 1:length(kmcdist) 
  kcaptcoutlong_tv   = YEAR(2).KCAPout((YEAR(2).kmc_silhouette{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KCAPTCoutlong_tv   = [KCAPTCoutlong_tv; kcaptcoutlong_tv];

  kscaptcoutlong_tv  = YEAR(2).KSCAPout((YEAR(2).kmcs_silhouette{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KSCAPTCoutlong_tv  = [KSCAPTCoutlong_tv; kscaptcoutlong_tv];

  kncaptcoutlong_tv  = YEAR(2).KCAPout((YEAR(2).kmcn_silhouette{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KNCAPTCoutlong_tv  = [KNCAPTCoutlong_tv; kncaptcoutlong_tv];
    
  kmc_num_sil   = YEAR(2).kmc_silhouette{1,i}.OptimalK;
  KMC_NUM_SIL   = [KMC_NUM_SIL; kmc_num_sil];

  kmcn_num_sil   = YEAR(2).kmcn_silhouette{1,i}.OptimalK;
  KMCN_NUM_SIL   = [KMCN_NUM_SIL; kmcn_num_sil];

  kmcs_num_sil   = YEAR(2).kmcs_silhouette{1,i}.OptimalK;
  KMCS_NUM_SIL   = [KMCS_NUM_SIL; kmcs_num_sil];
end

    FCAPTCoutlong_tv   = YEAR(2).FCAPout((YEAR(2).fcm_silhouette.OptimalK-1),2:end);
    FSCAPTCoutlong_tv  = YEAR(2).FSCAPout((YEAR(2).fcms_silhouette.OptimalK-1),2:end);
    FNCAPTCoutlong_tv  = YEAR(2).FNCAPout((YEAR(2).fcmn_silhouette.OptimalK-1),2:end);
    
    fcm_num_sil   = YEAR(2).fcm_silhouette.OptimalK;
    FCM_NUM_SIL   = [FCM_NUM_SIL, fcm_num_sil];

    fcmn_num_sil   = YEAR(2).fcmn_silhouette.OptimalK;
    FCMN_NUM_SIL   = [FCMN_NUM_SIL, fcmn_num_sil];

    fcms_num_sil   = YEAR(2).fcms_silhouette.OptimalK;
    FCMS_NUM_SIL   = [FCMS_NUM_SIL, fcms_num_sil];

for i = 1:length(cmcdist) 
  ccaptcoutlong_tv   = YEAR(2).CCAPout((YEAR(2).cmc_silhouette{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CCAPTCoutlong_tv   = [CCAPTCoutlong_tv; ccaptcoutlong_tv];

  cscaptcoutlong_tv  = YEAR(2).CSCAPout((YEAR(2).cmcs_silhouette{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CSCAPTCoutlong_tv  = [CSCAPTCoutlong_tv; cscaptcoutlong_tv];

  cncaptcoutlong_tv  = YEAR(2).CCAPout((YEAR(2).cmcn_silhouette{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CNCAPTCoutlong_tv  = [CNCAPTCoutlong_tv; cncaptcoutlong_tv];

  cmc_num_sil   = YEAR(2).cmc_silhouette{1,i}.OptimalK;
  CMC_NUM_SIL   = [CMC_NUM_SIL; cmc_num_sil];

  cmcn_num_sil   = YEAR(2).cmcn_silhouette{1,i}.OptimalK;
  CMCN_NUM_SIL   = [CMCN_NUM_SIL; cmcn_num_sil];

  cmcs_num_sil   = YEAR(2).cmcs_silhouette{1,i}.OptimalK;
  CMCS_NUM_SIL   = [CMCS_NUM_SIL; cmcs_num_sil];
end

for i = 1:length(hctdist)*length(hctalgo)
  hcaptcoutlong_tv   = YEAR(2).HCAPout((YEAR(2).hct_silhouette{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HCAPTCoutlong_tv   = [HCAPTCoutlong_tv; hcaptcoutlong_tv];

  hscaptcoutlong_tv  = YEAR(2).HSCAPout((YEAR(2).hcts_silhouette{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HSCAPTCoutlong_tv  = [HSCAPTCoutlong_tv; hscaptcoutlong_tv];

  hncaptcoutlong_tv  = YEAR(2).HCAPout((YEAR(2).hctn_silhouette{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HNCAPTCoutlong_tv  = [HNCAPTCoutlong_tv; hncaptcoutlong_tv];

  hct_num_sil   = YEAR(2).hct_silhouette{1,i}.OptimalK;
  HCT_NUM_SIL   = [HCT_NUM_SIL; hct_num_sil];

  hctn_num_sil  = YEAR(2).hctn_silhouette{1,i}.OptimalK;
  HCTN_NUM_SIL  = [HCTN_NUM_SIL; hctn_num_sil];

  hcts_num_sil  = YEAR(2).hcts_silhouette{1,i}.OptimalK;
  HCTS_NUM_SIL  = [HCTS_NUM_SIL; hcts_num_sil];
  
end

nshift = 0
for n = 3:18
    %k-means
INDCAPTCIT    = [];
NUM_SIL   = [];
parfor j = 1:size(KCAPTCoutlong_tv,1)
            IND = [];
            IND = price2ret(YEAR(n).KCAPout((YEAR(n).kmc_silhouette{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmc_silhouette{1,j}.OptimalK;
            INDCAPTC    = []; 
            ishift    = 0;
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));

           end
           INDCAPTCIT    = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
 end 
 KCAPTCoutlong_tv = [KCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
 KMC_NUM_SIL      = [KMC_NUM_SIL, NUM_SIL];

 INDCAPTCIT    = [];
 NUM_SIL  = [];
parfor j = 1:size(KSCAPTCoutlong_tv,1)
            IND = [];
            IND = price2ret(YEAR(n).KSCAPout((YEAR(n).kmcs_silhouette{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmcs_silhouette{1,j}.OptimalK;
            INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KSCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 

           NUM_SIL   = [NUM_SIL; num_sil];
 end 
 KSCAPTCoutlong_tv = [KSCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
 KMCS_NUM_SIL      = [KMCS_NUM_SIL, NUM_SIL];
 
 INDCAPTCIT    = [];
 NUM_SIL  = [];
parfor j = 1:size(KNCAPTCoutlong_tv,1)
            IND = [];
            IND = price2ret(YEAR(n).KNCAPout((YEAR(n).kmcn_silhouette{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmcn_silhouette{1,j}.OptimalK;
           INDCAPTC    = []; 
           ishift    = 0;
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KNCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
 end 
 KNCAPTCoutlong_tv = [KNCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
 KMCN_NUM_SIL      = [KMCN_NUM_SIL, NUM_SIL];
 
% FUZZY clustering
    INDCAPTCIT    = [];
    NUM_SIL   = [];
    parfor j = 1:size(FCAPTCoutlong_tv,1)
           IND = [];
           IND =  price2ret(YEAR(n).FCAPout((YEAR(n).fcm_silhouette.OptimalK-1), :),[], 'Periodic');
           num_sil   = YEAR(n).fcm_silhouette.OptimalK;
           INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
    end 
    FCAPTCoutlong_tv = [FCAPTCoutlong_tv, INDCAPTCIT(:,2:end)];
    FCM_NUM_SIL      = [FCM_NUM_SIL, NUM_SIL];
 
    INDCAPTCIT    = [];
    NUM_SIL   = [];      
    parfor j = 1:size(FNCAPTCoutlong_tv,1)
         
         IND = [];
         IND = price2ret(YEAR(n).FSCAPout((YEAR(n).fcms_silhouette.OptimalK-1),1:end),[], 'Periodic');
         num_sil   = YEAR(n).fcms_silhouette.OptimalK;
         INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FNCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
    end 
    FNCAPTCoutlong_tv = [FNCAPTCoutlong_tv, INDCAPTCIT(:,2:end)];  
    FCMN_NUM_SIL      = [FCMN_NUM_SIL, NUM_SIL];

    INDCAPTCIT    = [];
    NUM_SIL   = []; 
    parfor j = 1:size(FSCAPTCoutlong_tv,1)
         
         IND         = [];
         IND         = price2ret(YEAR(n).FNCAPout((YEAR(n).fcmn_silhouette.OptimalK-1),1:end),[], 'Periodic');
         num_sil     = YEAR(n).fcmn_silhouette.OptimalK;
         INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FSCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
    end 
    FSCAPTCoutlong_tv = [FSCAPTCoutlong_tv, INDCAPTCIT(:,2:end)];  
    FCMS_NUM_SIL      = [FCMS_NUM_SIL, NUM_SIL];


% c-medoids

INDCAPTCIT    = [];
NUM_SIL   = [];
parfor j = 1:size(CCAPTCoutlong_tv,1)
            IND = [];
            IND = price2ret(YEAR(n).CCAPout((YEAR(n).cmc_silhouette{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmc_silhouette{1,j}.OptimalK;            
            INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
NUM_SIL   = [NUM_SIL; num_sil];
           
 end 
CCAPTCoutlong_tv = [CCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
CMC_NUM_SIL      = [CMC_NUM_SIL, NUM_SIL];

INDCAPTCIT    = [];
NUM_SIL   = [];
parfor j = 1:size(CSCAPTCoutlong_tv,1)
            IND = [];
            IND = price2ret(YEAR(n).CSCAPout((YEAR(n).cmcs_silhouette{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmcs_silhouette{1,j}.OptimalK;            
            INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CSCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
            NUM_SIL   = [NUM_SIL; num_sil];
           
 end 
CSCAPTCoutlong_tv = [CSCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
CMCS_NUM_SIL      = [CMCS_NUM_SIL, NUM_SIL];

INDCAPTCIT    = [];
NUM_SIL   = [];
parfor j = 1:size(CNCAPTCoutlong_tv,1)
            IND = [];
            IND = price2ret(YEAR(n).CNCAPout((YEAR(n).cmcn_silhouette{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmcn_silhouette{1,j}.OptimalK;            
            INDCAPTC  = []; 
               
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CNCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));             
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
           
 end 
CNCAPTCoutlong_tv = [CNCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
CMCN_NUM_SIL      = [CMCN_NUM_SIL, NUM_SIL];


% Agglomerative clustering

INDCAPTCIT    = [];
NUM_SIL   = [];
parfor j = 1:size(HCAPTCoutlong_tv,1)
        IND = [];
        IND = price2ret(YEAR(n).HCAPout((YEAR(n).hct_silhouette{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
        INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
           
end 
HCAPTCoutlong_tv = [HCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
HCT_NUM_SIL      = [HCT_NUM_SIL, NUM_SIL];

INDCAPTCIT    = [];
NUM_SIL   = [];
parfor j = 1:size(HSCAPTCoutlong_tv,1)
            IND = [];
            IND = price2ret(YEAR(n).HSCAPout((YEAR(n).hcts_silhouette{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).hcts_silhouette{1,j}.OptimalK; 
            INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HSCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
           
 end 
HSCAPTCoutlong_tv = [HSCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
HCTS_NUM_SIL      = [HCTS_NUM_SIL, NUM_SIL];

INDCAPTCIT    = [];
NUM_SIL   = [];
parfor j = 1:size(HNCAPTCoutlong_tv,1)
            IND = [];
            IND = price2ret(YEAR(n).HNCAPout((YEAR(n).hctn_silhouette{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).hctn_silhouette{1,j}.OptimalK;
            INDCAPTC    = []; 
        
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HNCAPTCoutlong_tv(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_SIL   = [NUM_SIL; num_sil];
           
end 
HNCAPTCoutlong_tv = [HNCAPTCoutlong_tv, INDCAPTCIT(:,1:end)];
HCTN_NUM_SIL      = [HCTN_NUM_SIL, NUM_SIL];

nshift = nshift + 1 
end


%% Time-varying number of clusters construction of returns of max Sharpe portfolios: CalinskiHarabasz criterion
%based on absolute values for the whole period effeective number of clustering due to CalinskiHarabasz criterion
tic
KCAPTCoutlong_tv_CH  = [];
KSCAPTCoutlong_tv_CH = [];
KNCAPTCoutlong_tv_CH = [];

FCAPTCoutlong_tv_CH  = [];
FSCAPTCoutlong_tv_CH = [];
FNCAPTCoutlong_tv_CH = [];

CCAPTCoutlong_tv_CH  = [];
CSCAPTCoutlong_tv_CH = [];
CNCAPTCoutlong_tv_CH = [];

HCAPTCoutlong_tv_CH  = [];
HSCAPTCoutlong_tv_CH = [];
HNCAPTCoutlong_tv_CH = [];

KMC_NUM_CH = [];
KMCN_NUM_CH = [];
KMCS_NUM_CH = [];

FCM_NUM_CH = [];
FCMN_NUM_CH = [];
FCMS_NUM_CH = [];

CMC_NUM_CH = [];
CMCN_NUM_CH = [];
CMCS_NUM_CH = [];

HCT_NUM_CH = [];
HCTN_NUM_CH = [];
HCTS_NUM_CH = [];


for i = 1:length(kmcdist) 
  kcaptcoutlong_tv   = YEAR(2).KCAPout((YEAR(2).kmc_CalinskiHarabasz{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KCAPTCoutlong_tv_CH   = [KCAPTCoutlong_tv_CH; kcaptcoutlong_tv];

  kscaptcoutlong_tv  = YEAR(2).KSCAPout((YEAR(2).kmcs_CalinskiHarabasz{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KSCAPTCoutlong_tv_CH  = [KSCAPTCoutlong_tv_CH; kscaptcoutlong_tv];

  kncaptcoutlong_tv  = YEAR(2).KCAPout((YEAR(2).kmcn_CalinskiHarabasz{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KNCAPTCoutlong_tv_CH  = [KNCAPTCoutlong_tv_CH; kncaptcoutlong_tv];
    
  kmc_num_sil   = YEAR(2).kmc_CalinskiHarabasz{1,i}.OptimalK;
  KMC_NUM_CH   = [KMC_NUM_CH; kmc_num_sil];

  kmcn_num_sil   = YEAR(2).kmcn_CalinskiHarabasz{1,i}.OptimalK;
  KMCN_NUM_CH   = [KMCN_NUM_CH; kmcn_num_sil];

  kmcs_num_sil   = YEAR(2).kmcs_CalinskiHarabasz{1,i}.OptimalK;
  KMCS_NUM_CH   = [KMCS_NUM_CH; kmcs_num_sil];
end

    FCAPTCoutlong_tv_CH   = YEAR(2).FCAPout((YEAR(2).fcm_CalinskiHarabasz.OptimalK-1),2:end);
    FSCAPTCoutlong_tv_CH  = YEAR(2).FSCAPout((YEAR(2).fcms_CalinskiHarabasz.OptimalK-1),2:end);
    FNCAPTCoutlong_tv_CH  = YEAR(2).FNCAPout((YEAR(2).fcmn_CalinskiHarabasz.OptimalK-1),2:end);
    
    fcm_num_ch   = YEAR(2).fcm_CalinskiHarabasz.OptimalK;
    FCM_NUM_CH   = [FCM_NUM_CH, fcm_num_ch];

    fcmn_num_ch  = YEAR(2).fcmn_CalinskiHarabasz.OptimalK;
    FCMN_NUM_CH  = [FCMN_NUM_CH, fcmn_num_ch];

    fcms_num_ch  = YEAR(2).fcms_CalinskiHarabasz.OptimalK;
    FCMS_NUM_CH  = [FCMS_NUM_CH, fcms_num_ch];
 
for i = 1:length(cmcdist) 
  ccaptcoutlong_tv   = YEAR(2).CCAPout((YEAR(2).cmc_CalinskiHarabasz{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CCAPTCoutlong_tv_CH   = [CCAPTCoutlong_tv_CH; ccaptcoutlong_tv];

  cscaptcoutlong_tv  = YEAR(2).CSCAPout((YEAR(2).cmcs_CalinskiHarabasz{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CSCAPTCoutlong_tv_CH  = [CSCAPTCoutlong_tv_CH; cscaptcoutlong_tv];

  cncaptcoutlong_tv  = YEAR(2).CCAPout((YEAR(2).cmcn_CalinskiHarabasz{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CNCAPTCoutlong_tv_CH  = [CNCAPTCoutlong_tv_CH; cncaptcoutlong_tv];

  cmc_num_sil   = YEAR(2).cmc_CalinskiHarabasz{1,i}.OptimalK;
  CMC_NUM_CH   = [CMC_NUM_CH; cmc_num_sil];

  cmcn_num_sil   = YEAR(2).cmcn_CalinskiHarabasz{1,i}.OptimalK;
  CMCN_NUM_CH   = [CMCN_NUM_CH; cmcn_num_sil];

  cmcs_num_sil   = YEAR(2).cmcs_CalinskiHarabasz{1,i}.OptimalK;
  CMCS_NUM_CH   = [CMCS_NUM_CH; cmcs_num_sil];
end

for i = 1:length(hctdist)*length(hctalgo)
  hcaptcoutlong_tv   = YEAR(2).HCAPout((YEAR(2).hct_CalinskiHarabasz{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HCAPTCoutlong_tv_CH   = [HCAPTCoutlong_tv_CH; hcaptcoutlong_tv];

  hscaptcoutlong_tv  = YEAR(2).HSCAPout((YEAR(2).hcts_CalinskiHarabasz{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HSCAPTCoutlong_tv_CH  = [HSCAPTCoutlong_tv_CH; hscaptcoutlong_tv];

  hncaptcoutlong_tv  = YEAR(2).HCAPout((YEAR(2).hctn_CalinskiHarabasz{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HNCAPTCoutlong_tv_CH  = [HNCAPTCoutlong_tv_CH; hncaptcoutlong_tv];

  hct_num_sil   = YEAR(2).hct_CalinskiHarabasz{1,i}.OptimalK;
  HCT_NUM_CH   = [HCT_NUM_CH; hct_num_sil];

  hctn_num_sil  = YEAR(2).hctn_CalinskiHarabasz{1,i}.OptimalK;
  HCTN_NUM_CH  = [HCTN_NUM_CH; hctn_num_sil];

  hcts_num_sil  = YEAR(2).hcts_CalinskiHarabasz{1,i}.OptimalK;
  HCTS_NUM_CH  = [HCTS_NUM_CH; hcts_num_sil];
  
end

nshift = 0
for n = 3:18
    %k-means
INDCAPTCIT    = [];
NUM_CH   = [];
parfor j = 1:size(KCAPTCoutlong_tv_CH,1)
            IND = [];
            IND = price2ret(YEAR(n).KCAPout((YEAR(n).kmc_CalinskiHarabasz{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmc_CalinskiHarabasz{1,j}.OptimalK;
            INDCAPTC    = []; 
            ishift    = 0;
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT    = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
 end 
 KCAPTCoutlong_tv_CH = [KCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
 KMC_NUM_CH      = [KMC_NUM_CH, NUM_CH];

 INDCAPTCIT    = [];
 NUM_CH  = [];
parfor j = 1:size(KSCAPTCoutlong_tv_CH,1)
            IND = [];
            IND = price2ret(YEAR(n).KSCAPout((YEAR(n).kmcs_CalinskiHarabasz{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmcs_CalinskiHarabasz{1,j}.OptimalK;
            INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KSCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 

           NUM_CH   = [NUM_CH; num_sil];
 end 
 KSCAPTCoutlong_tv_CH = [KSCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
 KMCS_NUM_CH      = [KMCS_NUM_CH, NUM_CH];
 
 INDCAPTCIT    = [];
 NUM_CH  = [];
parfor j = 1:size(KNCAPTCoutlong_tv_CH,1)
            IND = [];
            IND = price2ret(YEAR(n).KNCAPout((YEAR(n).kmcn_CalinskiHarabasz{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmcn_CalinskiHarabasz{1,j}.OptimalK;
           INDCAPTC    = []; 
           ishift    = 0;
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KNCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
 end 
 KNCAPTCoutlong_tv_CH = [KNCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
 KMCN_NUM_CH      = [KMCN_NUM_CH, NUM_CH];

% FUZZY clustering

    INDCAPTCIT    = [];
    NUM_CH   = [];
    parfor j = 1:size(FCAPTCoutlong_tv_CH,1)
           IND = [];
           IND =  price2ret(YEAR(n).FCAPout((YEAR(n).fcm_CalinskiHarabasz.OptimalK-1), :),[], 'Periodic');
           num_sil   = YEAR(n).fcm_CalinskiHarabasz.OptimalK;
           INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
    end 
    FCAPTCoutlong_tv_CH = [FCAPTCoutlong_tv_CH, INDCAPTCIT(:,2:end)];
    FCM_NUM_CH      = [FCM_NUM_CH, NUM_CH];
 
    INDCAPTCIT    = [];
    NUM_CH   = [];      
    parfor j = 1:size(FNCAPTCoutlong_tv_CH,1)
         
         IND = [];
         IND = price2ret(YEAR(n).FSCAPout((YEAR(n).fcms_CalinskiHarabasz.OptimalK-1),1:end),[], 'Periodic');
         num_sil   = YEAR(n).fcms_CalinskiHarabasz.OptimalK;
         INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FNCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
    end 
    FNCAPTCoutlong_tv_CH = [FNCAPTCoutlong_tv_CH, INDCAPTCIT(:,2:end)];  
    FCMN_NUM_CH      = [FCMN_NUM_CH, NUM_CH];

    INDCAPTCIT    = [];
    NUM_CH   = []; 
    parfor j = 1:size(FSCAPTCoutlong_tv_CH,1)
         
         IND         = [];
         IND         = price2ret(YEAR(n).FNCAPout((YEAR(n).fcmn_CalinskiHarabasz.OptimalK-1),1:end),[], 'Periodic');
         num_sil     = YEAR(n).fcmn_CalinskiHarabasz.OptimalK;
         INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FSCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
    end 
    FSCAPTCoutlong_tv_CH = [FSCAPTCoutlong_tv_CH, INDCAPTCIT(:,2:end)];  
    FCMS_NUM_CH      = [FCMS_NUM_CH, NUM_CH];

% c-medoids

INDCAPTCIT    = [];
NUM_CH   = [];
parfor j = 1:size(CCAPTCoutlong_tv_CH,1)
            IND = [];
            IND = price2ret(YEAR(n).CCAPout((YEAR(n).cmc_CalinskiHarabasz{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmc_CalinskiHarabasz{1,j}.OptimalK;            
            INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
NUM_CH   = [NUM_CH; num_sil];
           
 end 
CCAPTCoutlong_tv_CH = [CCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
CMC_NUM_CH      = [CMC_NUM_CH, NUM_CH];

INDCAPTCIT    = [];
NUM_CH   = [];
parfor j = 1:size(CSCAPTCoutlong_tv_CH,1)
            IND = [];
            IND = price2ret(YEAR(n).CSCAPout((YEAR(n).cmcs_CalinskiHarabasz{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmcs_CalinskiHarabasz{1,j}.OptimalK;            
            INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CSCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
            NUM_CH   = [NUM_CH; num_sil];
           
 end 
CSCAPTCoutlong_tv_CH = [CSCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
CMCS_NUM_CH      = [CMCS_NUM_CH, NUM_CH];

INDCAPTCIT    = [];
NUM_CH   = [];
parfor j = 1:size(CNCAPTCoutlong_tv_CH,1)
            IND = [];
            IND = price2ret(YEAR(n).CNCAPout((YEAR(n).cmcn_CalinskiHarabasz{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmcn_CalinskiHarabasz{1,j}.OptimalK;            
            INDCAPTC  = []; 
               
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CNCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));             
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
           
 end 
CNCAPTCoutlong_tv_CH = [CNCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
CMCN_NUM_CH      = [CMCN_NUM_CH, NUM_CH];


% Agglomerative clustering

INDCAPTCIT    = [];
NUM_CH   = [];
parfor j = 1:size(HCAPTCoutlong_tv_CH,1)
        IND = [];
        IND = price2ret(YEAR(n).HCAPout((YEAR(n).hct_CalinskiHarabasz{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
        INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
           
end 
HCAPTCoutlong_tv_CH = [HCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
HCT_NUM_CH      = [HCT_NUM_CH, NUM_CH];

INDCAPTCIT    = [];
NUM_CH   = [];
parfor j = 1:size(HSCAPTCoutlong_tv_CH,1)
            IND = [];
            IND = price2ret(YEAR(n).HSCAPout((YEAR(n).hcts_CalinskiHarabasz{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).hcts_CalinskiHarabasz{1,j}.OptimalK; 
            INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HSCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
           
 end 
HSCAPTCoutlong_tv_CH = [HSCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
HCTS_NUM_CH      = [HCTS_NUM_CH, NUM_CH];

INDCAPTCIT    = [];
NUM_CH   = [];
parfor j = 1:size(HNCAPTCoutlong_tv_CH,1)
            IND = [];
            IND = price2ret(YEAR(n).HNCAPout((YEAR(n).hctn_CalinskiHarabasz{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).hctn_CalinskiHarabasz{1,j}.OptimalK;
            INDCAPTC    = []; 
        
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HNCAPTCoutlong_tv_CH(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_CH   = [NUM_CH; num_sil];
           
end 
HNCAPTCoutlong_tv_CH = [HNCAPTCoutlong_tv_CH, INDCAPTCIT(:,1:end)];
HCTN_NUM_CH      = [HCTN_NUM_CH, NUM_CH];

nshift = nshift + 1 
end
toc
%% Time-varying number of clusters construction of returns of max Sharpe portfolios: DaviesBouldin criterion
%based on absolute values for the whole period effeective number of clustering due to DaviesBouldin criterion
tic
KCAPTCoutlong_tv_DB  = [];
KSCAPTCoutlong_tv_DB = [];
KNCAPTCoutlong_tv_DB = [];

FCAPTCoutlong_tv_DB  = [];
FSCAPTCoutlong_tv_DB = [];
FNCAPTCoutlong_tv_DB = [];

CCAPTCoutlong_tv_DB  = [];
CSCAPTCoutlong_tv_DB = [];
CNCAPTCoutlong_tv_DB = [];

HCAPTCoutlong_tv_DB  = [];
HSCAPTCoutlong_tv_DB = [];
HNCAPTCoutlong_tv_DB = [];

KMC_NUM_DB= [];
KMCN_NUM_DB= [];
KMCS_NUM_DB= [];

FCM_NUM_DB= [];
FCMN_NUM_DB= [];
FCMS_NUM_DB= [];

CMC_NUM_DB= [];
CMCN_NUM_DB= [];
CMCS_NUM_DB= [];

HCT_NUM_DB= [];
HCTN_NUM_DB= [];
HCTS_NUM_DB= [];


for i = 1:length(kmcdist) 
  kcaptcoutlong_tv   = YEAR(2).KCAPout((YEAR(2).kmc_DaviesBouldin{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KCAPTCoutlong_tv_DB   = [KCAPTCoutlong_tv_DB; kcaptcoutlong_tv];

  kscaptcoutlong_tv  = YEAR(2).KSCAPout((YEAR(2).kmcs_DaviesBouldin{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KSCAPTCoutlong_tv_DB  = [KSCAPTCoutlong_tv_DB; kscaptcoutlong_tv];

  kncaptcoutlong_tv  = YEAR(2).KCAPout((YEAR(2).kmcn_DaviesBouldin{1,i}.OptimalK-2)*length(kmcdist)+i,2:end);
  KNCAPTCoutlong_tv_DB  = [KNCAPTCoutlong_tv_DB; kncaptcoutlong_tv];
    
  kmc_num_sil   = YEAR(2).kmc_DaviesBouldin{1,i}.OptimalK;
  KMC_NUM_DB  = [KMC_NUM_DB; kmc_num_sil];

  kmcn_num_sil   = YEAR(2).kmcn_DaviesBouldin{1,i}.OptimalK;
  KMCN_NUM_DB  = [KMCN_NUM_DB; kmcn_num_sil];

  kmcs_num_sil   = YEAR(2).kmcs_DaviesBouldin{1,i}.OptimalK;
  KMCS_NUM_DB  = [KMCS_NUM_DB; kmcs_num_sil];
end

    FCAPTCoutlong_tv_DB   = YEAR(2).FCAPout((YEAR(2).fcm_DaviesBouldin.OptimalK-1),2:end);
    FSCAPTCoutlong_tv_DB  = YEAR(2).FSCAPout((YEAR(2).fcms_DaviesBouldin.OptimalK-1),2:end);
    FNCAPTCoutlong_tv_DB  = YEAR(2).FNCAPout((YEAR(2).fcmn_DaviesBouldin.OptimalK-1),2:end);
       
    fcm_num_db   = YEAR(2).fcm_DaviesBouldin.OptimalK;
    FCM_NUM_DB   = [FCM_NUM_DB, fcm_num_db];

    fcmn_num_db  = YEAR(2).fcmn_DaviesBouldin.OptimalK;
    FCMN_NUM_DB  = [FCMN_NUM_DB, fcmn_num_db];

    fcms_num_db  = YEAR(2).fcms_DaviesBouldin.OptimalK;
    FCMS_NUM_DB  = [FCMS_NUM_DB, fcms_num_db];
    
for i = 1:length(cmcdist) 
  ccaptcoutlong_tv   = YEAR(2).CCAPout((YEAR(2).cmc_DaviesBouldin{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CCAPTCoutlong_tv_DB   = [CCAPTCoutlong_tv_DB; ccaptcoutlong_tv];

  cscaptcoutlong_tv  = YEAR(2).CSCAPout((YEAR(2).cmcs_DaviesBouldin{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CSCAPTCoutlong_tv_DB  = [CSCAPTCoutlong_tv_DB; cscaptcoutlong_tv];

  cncaptcoutlong_tv  = YEAR(2).CCAPout((YEAR(2).cmcn_DaviesBouldin{1,i}.OptimalK-2)*length(cmcdist)+i,2:end);
  CNCAPTCoutlong_tv_DB  = [CNCAPTCoutlong_tv_DB; cncaptcoutlong_tv];

  cmc_num_sil   = YEAR(2).cmc_DaviesBouldin{1,i}.OptimalK;
  CMC_NUM_DB  = [CMC_NUM_DB; cmc_num_sil];

  cmcn_num_sil   = YEAR(2).cmcn_DaviesBouldin{1,i}.OptimalK;
  CMCN_NUM_DB  = [CMCN_NUM_DB; cmcn_num_sil];

  cmcs_num_sil   = YEAR(2).cmcs_DaviesBouldin{1,i}.OptimalK;
  CMCS_NUM_DB  = [CMCS_NUM_DB; cmcs_num_sil];
end

for i = 1:length(hctdist)*length(hctalgo)
  hcaptcoutlong_tv   = YEAR(2).HCAPout((YEAR(2).hct_DaviesBouldin{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HCAPTCoutlong_tv_DB   = [HCAPTCoutlong_tv_DB; hcaptcoutlong_tv];

  hscaptcoutlong_tv  = YEAR(2).HSCAPout((YEAR(2).hcts_DaviesBouldin{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HSCAPTCoutlong_tv_DB  = [HSCAPTCoutlong_tv_DB; hscaptcoutlong_tv];

  hncaptcoutlong_tv  = YEAR(2).HCAPout((YEAR(2).hctn_DaviesBouldin{1,i}.OptimalK-2)*length(hctdist)*length(hctalgo)+i,2:end);
  HNCAPTCoutlong_tv_DB  = [HNCAPTCoutlong_tv_DB; hncaptcoutlong_tv];

  hct_num_sil   = YEAR(2).hct_DaviesBouldin{1,i}.OptimalK;
  HCT_NUM_DB  = [HCT_NUM_DB; hct_num_sil];

  hctn_num_sil  = YEAR(2).hctn_DaviesBouldin{1,i}.OptimalK;
  HCTN_NUM_DB = [HCTN_NUM_DB; hctn_num_sil];

  hcts_num_sil  = YEAR(2).hcts_DaviesBouldin{1,i}.OptimalK;
  HCTS_NUM_DB = [HCTS_NUM_DB; hcts_num_sil];
  
end

nshift = 0
for n = 3:18
    %k-means
INDCAPTCIT    = [];
NUM_DB  = [];
parfor j = 1:size(KCAPTCoutlong_tv_DB,1)
            IND = [];
            IND = price2ret(YEAR(n).KCAPout((YEAR(n).kmc_DaviesBouldin{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmc_DaviesBouldin{1,j}.OptimalK;
            INDCAPTC    = []; 
            ishift    = 0;
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT    = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
 end 
 KCAPTCoutlong_tv_DB = [KCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
 KMC_NUM_DB     = [KMC_NUM_DB, NUM_DB];

 INDCAPTCIT    = [];
 NUM_DB = [];
parfor j = 1:size(KSCAPTCoutlong_tv_DB,1)
            IND = [];
            IND = price2ret(YEAR(n).KSCAPout((YEAR(n).kmcs_DaviesBouldin{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmcs_DaviesBouldin{1,j}.OptimalK;
            INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KSCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 

           NUM_DB  = [NUM_DB; num_sil];
 end 
 KSCAPTCoutlong_tv_DB = [KSCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
 KMCS_NUM_DB     = [KMCS_NUM_DB, NUM_DB];
 
 INDCAPTCIT    = [];
 NUM_DB = [];
parfor j = 1:size(KNCAPTCoutlong_tv_DB,1)
            IND = [];
            IND = price2ret(YEAR(n).KNCAPout((YEAR(n).kmcn_DaviesBouldin{1,j}.OptimalK-2)*length(kmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).kmcn_DaviesBouldin{1,j}.OptimalK;
           INDCAPTC    = []; 
           ishift    = 0;
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*KNCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
 end 
 KNCAPTCoutlong_tv_DB = [KNCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
 KMCN_NUM_DB     = [KMCN_NUM_DB, NUM_DB];
for n = 3:18 
% FUZZY clustering
    INDCAPTCIT    = [];
    NUM_DB  = [];
parfor j = 1:size(FCAPTCoutlong_tv_DB,1)
           IND = [];
           IND =  price2ret(YEAR(n).FCAPout((YEAR(n).fcm_DaviesBouldin.OptimalK-1), :),[], 'Periodic');
           num_sil   = YEAR(n).fcm_DaviesBouldin.OptimalK;
           INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
    end 
    FCAPTCoutlong_tv_DB = [FCAPTCoutlong_tv_DB, INDCAPTCIT(:,2:end)];
    FCM_NUM_DB     = [FCM_NUM_DB, NUM_DB];
 
    INDCAPTCIT    = [];
    NUM_DB  = [];      
parfor j = 1:size(FNCAPTCoutlong_tv_DB,1)
         
         IND = [];
         IND = price2ret(YEAR(n).FSCAPout((YEAR(n).fcms_DaviesBouldin.OptimalK-1),1:end),[], 'Periodic');
         num_sil   = YEAR(n).fcms_DaviesBouldin.OptimalK;
         INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FNCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
    end 
    FNCAPTCoutlong_tv_DB = [FNCAPTCoutlong_tv_DB, INDCAPTCIT(:,2:end)];  
    FCMN_NUM_DB     = [FCMN_NUM_DB, NUM_DB];

    INDCAPTCIT    = [];
    NUM_DB  = []; 
parfor j = 1:size(FSCAPTCoutlong_tv_DB,1)
         
         IND         = [];
         IND         = price2ret(YEAR(n).FNCAPout((YEAR(n).fcmn_DaviesBouldin.OptimalK-1),1:end),[], 'Periodic');
         num_sil     = YEAR(n).fcmn_DaviesBouldin.OptimalK;
         INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*FSCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
    end 
    FSCAPTCoutlong_tv_DB = [FSCAPTCoutlong_tv_DB, INDCAPTCIT(:,2:end)];  
    FCMS_NUM_DB     = [FCMS_NUM_DB, NUM_DB];
end
% c-medoids

INDCAPTCIT    = [];
NUM_DB  = [];
parfor j = 1:size(CCAPTCoutlong_tv_DB,1)
            IND = [];
            IND = price2ret(YEAR(n).CCAPout((YEAR(n).cmc_DaviesBouldin{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmc_DaviesBouldin{1,j}.OptimalK;            
            INDCAPTC    = []; 

           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
NUM_DB  = [NUM_DB; num_sil];
           
 end 
CCAPTCoutlong_tv_DB = [CCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
CMC_NUM_DB     = [CMC_NUM_DB, NUM_DB];

INDCAPTCIT    = [];
NUM_DB  = [];
parfor j = 1:size(CSCAPTCoutlong_tv_DB,1)
            IND = [];
            IND = price2ret(YEAR(n).CSCAPout((YEAR(n).cmcs_DaviesBouldin{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmcs_DaviesBouldin{1,j}.OptimalK;            
            INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CSCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
            NUM_DB  = [NUM_DB; num_sil];
           
 end 
CSCAPTCoutlong_tv_DB = [CSCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
CMCS_NUM_DB     = [CMCS_NUM_DB, NUM_DB];

INDCAPTCIT    = [];
NUM_DB  = [];
parfor j = 1:size(CNCAPTCoutlong_tv_DB,1)
            IND = [];
            IND = price2ret(YEAR(n).CNCAPout((YEAR(n).cmcn_DaviesBouldin{1,j}.OptimalK-2)*length(cmcdist)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).cmcn_DaviesBouldin{1,j}.OptimalK;            
            INDCAPTC  = []; 
               
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CNCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));             
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
           
 end 
CNCAPTCoutlong_tv_DB = [CNCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
CMCN_NUM_DB     = [CMCN_NUM_DB, NUM_DB];


% Agglomerative clustering

INDCAPTCIT    = [];
NUM_DB  = [];
parfor j = 1:size(HCAPTCoutlong_tv_DB,1)
        IND = [];
        IND = price2ret(YEAR(n).HCAPout((YEAR(n).hct_DaviesBouldin{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
        INDCAPTC    = []; 
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
           
end 
HCAPTCoutlong_tv_DB = [HCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
HCT_NUM_DB     = [HCT_NUM_DB, NUM_DB];

INDCAPTCIT    = [];
NUM_DB  = [];
parfor j = 1:size(HSCAPTCoutlong_tv_DB,1)
            IND = [];
            IND = price2ret(YEAR(n).HSCAPout((YEAR(n).hcts_DaviesBouldin{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).hcts_DaviesBouldin{1,j}.OptimalK; 
            INDCAPTC    = []; 
           
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HSCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
           
 end 
HSCAPTCoutlong_tv_DB = [HSCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
HCTS_NUM_DB     = [HCTS_NUM_DB, NUM_DB];

INDCAPTCIT    = [];
NUM_DB  = [];
parfor j = 1:size(HNCAPTCoutlong_tv_DB,1)
            IND = [];
            IND = price2ret(YEAR(n).HNCAPout((YEAR(n).hctn_DaviesBouldin{1,j}.OptimalK-2)*length(hctdist)*length(hctalgo)+j,2:end)',[], 'Periodic')';
            num_sil   = YEAR(n).hctn_DaviesBouldin{1,j}.OptimalK;
            INDCAPTC    = []; 
        
    
           for i = 1:size(IND,2)
               INDCAPTC(1) = tc*HNCAPTCoutlong_tv_DB(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
              

           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
           NUM_DB  = [NUM_DB; num_sil];
           
end 
HNCAPTCoutlong_tv_DB = [HNCAPTCoutlong_tv_DB, INDCAPTCIT(:,1:end)];
HCTN_NUM_DB     = [HCTN_NUM_DB, NUM_DB];

nshift = nshift + 1 
end
toc
%% Sort strategies by performance measure (Cum return)
%
%k-means
KMPARtr = [];%matrix of algorithms parameters in INDKMC  format
kmpar = {};
for j = 1:length(kmcdist)
for i = 2:clus_num
kmpar{i} = strcat(kmcdist(j),',  ', num2str(i))
end
KMPARtr = [KMPARtr; kmpar(:,2:end)];
end
%C-medoids
CMCPARtr = [];%matrix of algorithms parameters in INDCMC  format
cmcpar = {};
for j = 1:length(cmcdist)
for i = 2:clus_num
cmcpar{i} = strcat(cmcdist(j),',  ', num2str(i));
end
CMCPARtr = [CMCPARtr; cmcpar(:,2:end)];
end

% Agglomerative clusters
HCTPARtr = [];
hctpartr = {};
for j = 2:clus_num
HCTpar = [];
for n = 1:length(hctdist)
for i = 1:length(hctalgo)
hctpartr{i} = strcat(hctdist(n),',  ',hctalgo(i), ', ', num2str(j)); %matrix of algorithms parameters in INDHCT  format
end
HCTpar = [HCTpar, hctpartr];
end
HCTPARtr = [HCTPARtr, HCTpar'];
end

KMPAR     = [];
KCAPSORT  = [];

KSPAR     = [];
KSCAPSORT = [];

KNPAR     = [];
KNCAPSORT = [];

HPAR     = [];
HCAPSORT  = [];

HSPAR     = [];
HSCAPSORT = [];

HNPAR     = [];
HNCAPSORT = [];

CMPAR     = [];
CCAPSORT  = [];

CSPAR     = [];
CSCAPSORT = [];

CNPAR     = [];
CNCAPSORT = [];
% Sort final performance for 1/n portfolios
KPARsortlong               = [];
KSPARsortlong              = [];
KNPARsortlong              = [];

HPARsortlong               = [];
HSPARsortlong              = [];
HNPARsortlong              = [];

CPARsortlong               = [];
CSPARsortlong              = [];
CNPARsortlong              = [];

[KCAPTCsort, Kparlong]   = sort(KCAPTCoutlong(:,end),'descend');
[KSCAPTCsort, KSparlong] = sort(KSCAPTCoutlong(:,end),'descend');
[KNCAPTCsort, KNparlong ] = sort(KNCAPTCoutlong(:,end),'descend');

[CCAPTCsort, Cparlong]   = sort(CCAPTCoutlong(:,end),'descend');
[CSCAPTCsort, CSparlong] = sort(CSCAPTCoutlong(:,end),'descend');
[CNCAPTCsort, CNparlong] = sort(CNCAPTCoutlong(:,end),'descend');

[HCAPTCsort, Hparlong]   = sort(HCAPTCoutlong(:,end),'descend');
[HSCAPTCsort, HSparlong] = sort(HSCAPTCoutlong(:,end),'descend');
[HNCAPTCsort, HNparlong ] = sort(HNCAPTCoutlong(:,end),'descend');

%Best strategies parameters
CPARsortlong               = [CPARsortlong, [CMCPARtr{Cparlong(1:end)}]'];
CSPARsortlong              = [CSPARsortlong, [CMCPARtr{CSparlong(1:end)}]'];
CNPARsortlong              = [CNPARsortlong, [CMCPARtr{CNparlong(1:end)}]'];

KPARsortlong               = [KPARsortlong, [KMPARtr{Kparlong(1:end)}]'];
KSPARsortlong              = [KSPARsortlong, [KMPARtr{KSparlong(1:end)}]'];
KNPARsortlong              = [KNPARsortlong, [KMPARtr{KNparlong(1:end)}]'];

HPARsortlong               = [HPARsortlong, [HCTPARtr{Hparlong(1:end)}]'];
HSPARsortlong              = [HSPARsortlong, [HCTPARtr{HSparlong(1:end)}]'];
HNPARsortlong              = [HNPARsortlong, [HCTPARtr{HNparlong(1:end)}]'];

%Sort 1/n annual performance
for n = 2:length(RET_YEAR)-1
% K-mean
clear Kpar KSpar KNpar KCAPsort KSCAPsort KNCAPsort
[KCAPsort, Kpar]   = sort(YEAR(n).KCAPout(:,end),'descend');
[KSCAPsort, KSpar] = sort(YEAR(n).KSCAPout(:,end),'descend');
[KNCAPsort, KNpar] = sort(YEAR(n).KNCAPout(:,end),'descend');
KMPAR              = [KMPAR, [KMPARtr{Kpar(1:end)}]'];
KCAPSORT           = [KCAPSORT, KCAPsort];
KSPAR              = [KSPAR, [KMPARtr{KSpar(1:end)}]'];
KSCAPSORT          = [KSCAPSORT, KSCAPsort];
KNPAR              = [KNPAR, [KMPARtr{KNpar(1:end)}]'];
KNCAPSORT          = [KNCAPSORT, KNCAPsort];

%C-medoids
clear Cpar CSpar CNpar CCAPsort CSCAPsort CNCAPsort
[CCAPsort, Cpar]   = sort(YEAR(n).CCAPout(:,end),'descend');
[CSCAPsort, CSpar] = sort(YEAR(n).CSCAPout(:,end),'descend');
[CNCAPsort, CNpar] = sort(YEAR(n).CNCAPout(:,end),'descend');
CMPAR              = [CMPAR, [CMCPARtr{Cpar(1:end)}]'];
CCAPSORT           = [CCAPSORT, CCAPsort];
CSPAR              = [CSPAR, [CMCPARtr{CSpar(1:end)}]'];
CSCAPSORT          = [CSCAPSORT, CSCAPsort];
CNPAR              = [CNPAR, [CMCPARtr{CNpar(1:end)}]'];
CNCAPSORT          = [CNCAPSORT, CNCAPsort];

%Agglomerative
clear Hpar HSpar HNpar HCAPsort HSCAPsort HNCAPsort
[HCAPsort, Hpar]   = sort(YEAR(n).HCAPout(:,end),'descend');
[HSCAPsort, HSpar] = sort(YEAR(n).HSCAPout(:,end),'descend');
[HNCAPsort, HNpar] = sort(YEAR(n).HNCAPout(:,end),'descend');
HPAR               = [HPAR, [HCTPARtr{Hpar(1:end)}]'];
HCAPSORT           = [HCAPSORT, HCAPsort];
HSPAR              = [HSPAR, [HCTPARtr{HSpar(1:end)}]'];
HSCAPSORT          = [HSCAPSORT, HSCAPsort];
HNPAR              = [HNPAR, [HCTPARtr{HNpar(1:end)}]'];
HNCAPSORT          = [HNCAPSORT, HNCAPsort];
end

disp 'Top-10 Agglomerative strategies'
HPARlong  = [HCTPARtr{Hparlong(1:end)}]';
HSPARlong = [HCTPARtr{HSparlong(1:end)}]';
HNPARlong = [HCTPARtr{HNparlong(1:end)}]';
disp 'Top-10 Agglomerative strategies'
HPARlong  = [HCTPARtr{Hparlong(1:end)}]';
HSPARlong = [HCTPARtr{HSparlong(1:end)}]';
HNPARlong = [HCTPARtr{HNparlong(1:end)}]';
disp 'Top-10 Agglomerative strategies'
HPARlong  = [HCTPARtr{Hparlong(1:10)}]';
HSPARlong = [HCTPARtr{HSparlong(1:10)}]';
HNPARlong = [HCTPARtr{HNparlong(1:10)}]';

% Sort final performance for MV portfolios
KMVPARsortlong               = [];
KSMVPARsortlong              = [];
KNMVPARsortlong              = [];

HMVPARsortlong               = [];
HSMVPARsortlong              = [];
HNMVPARsortlong              = [];

CMVPARsortlong               = [];
CSMVPARsortlong              = [];
CNMVPARsortlong              = [];

[KMVCAPTCsort, KMVparlong]   = sort(KMVCAPTCoutlong(:,end),'descend');
[KSMVCAPTCsort, KSMVparlong] = sort(KSMVCAPTCoutlong(:,end),'descend');
[KNMVCAPTCsort, KNMVparlong ] = sort(KNMVCAPTCoutlong(:,end),'descend');

[CMVCAPTCsort, CMVparlong]   = sort(CMVCAPTCoutlong(:,end),'descend');
[CSMVCAPTCsort, CSMVparlong] = sort(CSMVCAPTCoutlong(:,end),'descend');
[CNMVCAPTCsort, CNMVparlong] = sort(CNMVCAPTCoutlong(:,end),'descend');

[HMVCAPTCsort, HMVparlong]   = sort(HMVCAPTCoutlong(:,end),'descend');
[HSMVCAPTCsort, HSMVparlong] = sort(HSMVCAPTCoutlong(:,end),'descend');
[HNMVCAPTCsort, HNMVparlong ] = sort(HNMVCAPTCoutlong(:,end),'descend');

%Best strategies parameters
CMVPARsortlong               = [CMVPARsortlong, [CMCPARtr{CMVparlong(1:end)}]'];
CSMVPARsortlong              = [CSMVPARsortlong, [CMCPARtr{CSMVparlong(1:end)}]'];
CNMVPARsortlong              = [CNMVPARsortlong, [CMCPARtr{CNMVparlong(1:end)}]'];

KMVPARsortlong               = [KMVPARsortlong, [KMPARtr{KMVparlong(1:end)}]'];
KSMVPARsortlong              = [KSMVPARsortlong, [KMPARtr{KSMVparlong(1:end)}]'];
KNMVPARsortlong              = [KNMVPARsortlong, [KMPARtr{KNMVparlong(1:end)}]'];

HMVPARsortlong               = [HMVPARsortlong, [HCTPARtr{HMVparlong(1:end)}]'];
HSMVPARsortlong              = [HSMVPARsortlong, [HCTPARtr{HSMVparlong(1:end)}]'];
HNMVPARsortlong              = [HNMVPARsortlong, [HCTPARtr{HNMVparlong(1:end)}]'];
%Sort annual MV portfolios performance
KMVPAR     = [];
KMVCAPSORT  = [];

KSMVPAR     = [];
KSMVCAPSORT = [];

KNMVPAR     = [];
KNMVCAPSORT = [];

HMVPAR     = [];
HMVCAPSORT  = [];

HSMVPAR     = [];
HSMVCAPSORT = [];

HNMVPAR     = [];
HNMVCAPSORT = [];

CMVPAR     = [];
CMVCAPSORT  = [];

CSMVPAR     = [];
CSMVCAPSORT = [];

CNMVPAR     = [];
CNMVCAPSORT = [];

for n = 2:length(RET_YEAR)-1
% KMV-mean
clear KMVpar KSMVpar KNMVpar KMVCAPsort KSMVCAPsort KNMVCAPsort
[KMVCAPsort, KMVpar]   = sort(YEAR(n).KMVCAPout(:,end),'descend');
[KSMVCAPsort, KSMVpar] = sort(YEAR(n).KSMVCAPout(:,end),'descend');
[KNMVCAPsort, KNMVpar] = sort(YEAR(n).KNMVCAPout(:,end),'descend');
KMVPAR               = [KMVPAR, [KMPARtr{KMVpar(1:end)}]'];
KMVCAPSORT           = [KMVCAPSORT, KMVCAPsort];
KSMVPAR              = [KSMVPAR, [KMPARtr{KSMVpar(1:end)}]'];
KSMVCAPSORT          = [KSMVCAPSORT, KSMVCAPsort];
KNMVPAR              = [KNMVPAR, [KMPARtr{KNMVpar(1:end)}]'];
KNMVCAPSORT          = [KNMVCAPSORT, KNMVCAPsort];

%C-medoids
clear CMVpar CSMVpar CNMVpar CMVCAPsort CSMVCAPsort CNMVCAPsort
[CMVCAPsort, CMVpar]   = sort(YEAR(n).CMVCAPout(:,end),'descend');
[CSMVCAPsort, CSMVpar] = sort(YEAR(n).CSMVCAPout(:,end),'descend');
[CNMVCAPsort, CNMVpar] = sort(YEAR(n).CNMVCAPout(:,end),'descend');
CMVPAR               = [CMVPAR, [CMCPARtr{CMVpar(1:end)}]'];
CMVCAPSORT           = [CMVCAPSORT, CMVCAPsort];
CSMVPAR              = [CSMVPAR, [CMCPARtr{CSMVpar(1:end)}]'];
CSMVCAPSORT          = [CSMVCAPSORT, CSMVCAPsort];
CNMVPAR              = [CNMVPAR, [CMCPARtr{CNMVpar(1:end)}]'];
CNMVCAPSORT          = [CNMVCAPSORT, CNMVCAPsort];

%Agglomerative
clear HMVpar HSMVpar HNMVpar HMVCAPsort HSMVCAPsort HNMVCAPsort
[HMVCAPsort, HMVpar]   = sort(YEAR(n).HMVCAPout(:,end),'descend');
[HSMVCAPsort, HSMVpar] = sort(YEAR(n).HSMVCAPout(:,end),'descend');
[HNMVCAPsort, HNMVpar] = sort(YEAR(n).HNMVCAPout(:,end),'descend');
HMVPAR               = [HMVPAR, [HCTPARtr{HMVpar(1:end)}]'];
HMVCAPSORT           = [HMVCAPSORT, HMVCAPsort];
HSMVPAR              = [HSMVPAR, [HCTPARtr{HSMVpar(1:end)}]'];
HSMVCAPSORT          = [HSMVCAPSORT, HSMVCAPsort];
HNMVPAR              = [HNMVPAR, [HCTPARtr{HNMVpar(1:end)}]'];
HNMVCAPSORT          = [HNMVCAPSORT, HNMVCAPsort];
end
%% MV CUMULATIVE RETURN CALCULATION LOOP
WGT           = {};
COVMOG        = {};
VaROG         = [];
num_dig       = 4;
TargRet       = 0.8;
NUMFAC        = [];
MVCAPRETlong  = [];
options       = optimset('Algorithm','active-set','MaxFunEvals',100000);
for n = 2:length(RET_YEAR)-1
    clear capit OGCAPIT
    MVCAPRET = [];
    OGCAPIT       = [];
    capit(1)      = 1;
    OGRETI   = YEAR(n).DATA(:,1:end-1);  %
    Data_old = YEAR(n-1).DATA(:,1:end-1);%last coulumn is an index
    Tick          = YEAR(n).TICK;
    Tick_old      = YEAR(n-1).TICK; 
    ind           = find(ismember(Tick_old, Tick));

        XI               = Data_old;
        %[Ht,numfactors] = ogarch(XI,1,1,1,1,varthresh); %calculating the time-varying covariance matrix via the orthogonal GARCH
        %setting up the variance-covariance VaR optimization procedure
        Ht              = cov(XI); 
        w0              = ones(1,size(XI,2))./size(XI,2);
        MeanRet         = mean(XI)';
        ub              = ones(length(w0),1);
        lb              = zeros(length(w0),1);
        Aeq             = ones(1,length(w0));
        beq             = 1;
        AA              = -MeanRet';
        bb              = -quantile(MeanRet,TargRet);
        [wgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        wgt             = round(wgt.*(10^num_dig))./(10^num_dig);
        
  for i = 1:size(OGRETI,1);
        %portfolio appreciation
        capit(i+1)      = sum(capit(i).*wgt'.*((1 + OGRETI(i,ind))'));
        caplast         = capit(end);
        OGCAPIT         = [OGCAPIT,caplast];
      
    end
    VaROG   = [VaROG,iVaR]; %calculating the VaR vector
   
    OGCAPITST = [1,OGCAPIT]; %final cumulative return vector for Strategy 4 with transaction costs
    MVCAPRET  = [MVCAPRET; price2ret(OGCAPITST,[], 'Periodic')];
    MVCAPRETlong     = [MVCAPRETlong, MVCAPRET];
    
    YEAR(n).MVCAP    =  OGCAPITST;
    YEAR(n).MVWGT    =  wgt;
    YEAR(n).MVCAPRET =  MVCAPRET;  
end

% Construction of long general MV portfolio cumulative returns with TC
nshift = 0
tic
    MVCAPTClong  = YEAR(2).MVCAP(:,2:end);
   
for n = 3:length(RET_YEAR)-1
    INDCAPTCIT    = [];
    for j = 1:size(MVCAPRETlong,1)
            IND = [];
            IND = YEAR(n).MVCAPRET(j,:); 
            INDCAPTC    = []; 
  
            for i = 1:size(IND,2)
               INDCAPTC(1) = tc*MVCAPTClong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
          
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    MVCAPTClong = [MVCAPTClong, INDCAPTCIT(:,2:end)]; 

nshift = nshift + 1 
end 

toc %ca 

%%  CUMULATIVE 1/n RETURN for all stocks rebalanced annualy
CAPRETlong     = [];

nshift = 0 
for n = 2:length(RET_YEAR)-1
    Data       = YEAR(n).DATA;
    CAPoutall  = []; 
    CAPRET     = [];

    clear cap capp cappp  

    CAPout  = []; 
    cappp   = [];
    cap{1}  = 1; 
    capp(1) = 1;


    for l = 1:size(Data,1)  
    
        cap{l}   = (repmat(sum(cell2mat(cap(l)))/size(Data,2),1,size(Data,2)));
        cap{l+1} = sum(cell2mat(cap(l)).*(1 + Data(l,:)));
        capp     = cap{l+1};
        cappp    = [cappp,capp];  
    end     
    CAPout      = [1, cappp]; 
    CAPRET      = [CAPRET; price2ret(CAPout,[], 'Periodic')];
    CAPoutall   = [CAPoutall; CAPout]; 

CAPRETlong    = [CAPRETlong, CAPRET];

YEAR(n).CAPRET  = CAPRET;
YEAR(n).CAPout  = CAPoutall;
end 

CAPTClong  = YEAR(2).CAPout(:,2:end);
   
for n = 3:length(RET_YEAR)-1
    INDCAPTCIT    = [];
    for j = 1:size(CAPRETlong,1)
            IND = [];
            IND = YEAR(n).CAPRET(j,:); 
            INDCAPTC    = []; 
  
            for i = 1:size(IND,2)
               INDCAPTC(1) = tc*CAPTClong(j,end); 
               INDCAPTC(i+1) = INDCAPTC(i)*(1+IND(i));
          
           end
           INDCAPTCIT  = [INDCAPTCIT; INDCAPTC]; 
    end 
    CAPTClong = [CAPTClong, INDCAPTCIT(:,2:end)]; 

nshift = nshift + 1 
end 
toc %ca 
%% Latex table
KCAPOUTLONGtable = round(KCAPoutlong(:,end),4);
openvar('KCAPOUTLONGtable')

FCAPOUTLONGtable = round(FCAPoutlong(:,end),4);
openvar('FCAPOUTLONGtable')

HCAPOUTLONGtable = round(HCAPoutlong(:,end),4);
openvar('HCAPOUTLONGtable')

KMVCAPOUTLONGtable = round(KMVCAPoutlong(:,end),4);
openvar('KMVCAPOUTLONGtable')

FMVCAPOUTLONGtable = round(FMVCAPoutlong(:,end),4);
openvar('FMVCAPOUTLONGtable')

HMVCAPOUTLONGtable = round(HMVCAPoutlong(:,end),4);
openvar('HMVCAPOUTLONGtable')
%For 1/n portfolios
disp ('k-means Top 10 strtegies')
[KMPARtr{Kpar(1:10)}]'
[KMPARtr{KSpar(1:10)}]' 
[KMPARtr{KNpar(1:10)}]'
input.data                      = [round(KCAPTCsort(1:10),2), zeros(10,1), round(KSCAPTCsort(1:10),2),zeros(10,1), round(KNCAPTCsort(1:10),2), zeros(10,1)];
 %[round(KCAPTCsort(1:10),2), {KMPAR(Kpar(1:10))'}, {round(KSCAPTCsort(1:10),2)}, {KMPAR(KSpar(1:10))'}, {round(KNCAPTCsort(1:10),2)}, {KMPAR(KNpar(1:10))'}];
input.tableColLabels            = {'Top-10 row data', 'Parameters'...
                                   'Top-10 std data', 'Parameters'...
                                   'Top-10 norm data', 'Parameters'};
%input.tableRowLabels            = {'3 ', '4 ', '5 ','6', '7', '8 ', '9', '10', '11', 
      %                             '12'};
input.transposeTable            = 0;
input.dataFormatMode            = 'column'; 
input.dataNanString             = '-';
input.tableColumnAlignment      = 'c';
input.tableBorders              = 1;
input.tableCaption              = 'k-means 1/n portfolios performance';
input.tableLabel                = 'MyTableLabel';
input.makeCompleteLatexDocument = 1;
latex                           = latexTable(input);


%C-medoids Top 10 strtegies

disp ('k-means Top 10 strtegies')
[CMCPARtr{Cpar(1:10)}]'
[CMCPARtr{CSpar(1:10)}]' 
[CMCPARtr{CNpar(1:10)}]'
input.data                      = [round(CCAPTCsort(1:10),2), zeros(10,1), round(CSCAPTCsort(1:10),2),zeros(10,1), round(CNCAPTCsort(1:10),2), zeros(10,1)];
 %[round(KCAPTCsort(1:10),2), {KMPAR(Kpar(1:10))'}, {round(KSCAPTCsort(1:10),2)}, {KMPAR(KSpar(1:10))'}, {round(KNCAPTCsort(1:10),2)}, {KMPAR(KNpar(1:10))'}];
input.tableColLabels            = {'Top-10 row data', 'Parameters'...
                                   'Top-10 std data', 'Parameters'...
                                   'Top-10 norm data', 'Parameters'};
%input.tableRowLabels            = {'3 ', '4 ', '5 ','6', '7', '8 ', '9', '10', '11', 
      %                             '12'};
input.transposeTable            = 0;
input.dataFormatMode            = 'column'; 
input.dataNanString             = '-';
input.tableColumnAlignment      = 'c';
input.tableBorders              = 1;
input.tableCaption              = 'k-means 1/n portfolios performance';
input.tableLabel                = 'MyTableLabel';
input.makeCompleteLatexDocument = 1;
latex                           = latexTable(input);
%end

%Agglomerative clusters' Top 10 strtegies
input.data                      = [round(HCAPTCsort(1:10),2), zeros(10,1), round(HSCAPTCsort(1:10),2),zeros(10,1), round(HNCAPTCsort(1:10),2), zeros(10,1)];
 %[round(KCAPTCsort(1:10),2), {KMPAR(Kpar(1:10))'}, {round(KSCAPTCsort(1:10),2)}, {KMPAR(KSpar(1:10))'}, {round(KNCAPTCsort(1:10),2)}, {KMPAR(KNpar(1:10))'}];
input.tableColLabels            = {'Top-10 row data', 'Parameters'...
                                   'Top-10 std data', 'Parameters'...
                                   'Top-10 norm data', 'Parameters'};
%input.tableRowLabels            = {'3 ', '4 ', '5 ','6', '7', '8 ', '9', '10', '11', 
      %                             '12'};
input.transposeTable            = 0;
input.dataFormatMode            = 'column'; 
input.dataNanString             = '-';
input.tableColumnAlignment      = 'c';
input.tableBorders              = 1;
input.tableCaption              = 'k-means 1/n portfolios performance';
input.tableLabel                = 'MyTableLabel';
input.makeCompleteLatexDocument = 1;
latex                           = latexTable(input);
%end

%For MV portfolios
%for n = 1:3
input.data                      = [KMVCAPoutlong(:,end), FMVCAPoutlong(:,end),HMVCAPoutlong(:,end),repmat(INDCAPITlong(1,end),10,1)];

input.tableColLabels            = {'K-means','Fuzzy C-means',...
                                   'Agglomerative hierarchical clustering',...
                                   'STOXXUSA600'};
input.tableRowLabels            = {'3 ', '4 ', '5 ','6', '7', '8 ', '9', '10', '11', 
                                   '12'};
input.transposeTable            = 0;
input.dataFormatMode            = 'column'; 
input.dataNanString             = '-';
input.tableColumnAlignment      = 'c';
input.tableBorders              = 1;
input.tableCaption              = 'Markowitz-portfolios cumulative return';
input.tableLabel                = 'MyTableLabel';
input.makeCompleteLatexDocument = 1;
latex                           = latexTable(input);

%% Plots' parameters
% Create color sequence (Rainbow)

rgb = 255;

color1  = [1 0 0];
color2  = [238 82 3]/rgb;
color3  = [1 0.494117647409439 0];
color4  = [1 0.749019622802734 0];
color5  = [248 212 33]/rgb;
color6  = [0 1 0];
color7  = [33 225 8]/rgb;
color8  = [0 0.498039215803146 0];
color9  = [74 233 248]/rgb;
color10  = [0 0.498039215803146 1];
color11  = [0 0 1];
color12  = [0.0784313753247261 0.168627455830574 0.549019634723663];
color13  = [0.749019622802734 0 0.749019622802734];
color14 = [0.47843137383461 0.062745101749897 0.894117653369904];
rainbow14 = [color1; color2;color3;color4;color5;color6;color7;color8;color9;color10; color11; color12; color13; color14];

Datelong = [YEAR(2).Date;  YEAR(3).Date;  YEAR(4).Date;  YEAR(5).Date;  YEAR(6).Date;...
            YEAR(7).Date;  YEAR(8).Date;  YEAR(9).Date;  YEAR(10).Date; YEAR(11).Date;...
            YEAR(12).Date; YEAR(13).Date; YEAR(14).Date; YEAR(15).Date;...
            YEAR(16).Date; YEAR(17).Date; YEAR(18).Date]';
Datelong =  PRICE(263:end,1)+693960;
        
lineThick = 6
Fontsize = 20
%% Plots of time varying portfolios
figure
subplot(2,1,1)       % add first plot in 2 x 1 grid
plot(Datelong, KSCAPTCoutlong_tv(4,:),'Color','r','LineWidth',lineThick)
%xlabel('Time')
datetick('x','yyyy', 'keepticks')
ylabel('Portfolio Wealth')
title('1/n portfolios with time-varying number of clusters')
hold on
plot(Datelong,FSCAPTCoutlong_tv,'Color','b', 'LineWidth',lineThick)
plot(Datelong,HSCAPTCoutlong_tv(6,:),'Color',color5,'LineWidth',lineThick)
plot(Datelong,CSCAPTCoutlong_tv(7,:),'Color','m','LineWidth',lineThick)
hold off

subplot(2,1,2)       % add second plot in 2 x 1 grid
plot(Datelong2,KMCS_NUM_SIL(4,:),'*','MarkerSize',10,'Color','r') 

xlabel('Time')
ylabel('Number of clusters'), 
title('Number of clusters due to Silhouette criterion')
hold on
datetick('x','yyyy', 'keepticks')
plot(Datelong2,FCMS_NUM_SIL,'o','MarkerSize',10,'Color','b') 
plot(Datelong2,HCTS_NUM_SIL(6,:),'+','MarkerSize',10,'Color',color5) 
plot(Datelong2,CMCS_NUM_SIL(7,:),'x','MarkerSize',10,'Color','m') 
hold off
%%%%%%%%%%%%%%%%%
figure
subplot(2,1,1)       % add first plot in 2 x 1 grid
plot(Datelong, KSCAPTCoutlong_tv_CH(4,:),'Color','r','LineWidth',lineThick)
%xlabel('Time')
ylabel('Portfolio Wealth')
title('1/n portfolios with time-varying number of clusters')
ax = gca;
ax.XTickLabel = {};
hold on
plot(Datelong,FSCAPTCoutlong_tv_CH,'Color','b', 'LineWidth',lineThick)
plot(Datelong,HSCAPTCoutlong_tv_CH(6,:),'Color',color5,'LineWidth',lineThick)
plot(Datelong,CSCAPTCoutlong_tv_CH(7,:),'Color','m','LineWidth',lineThick)
hold off

subplot(2,1,2)       % add second plot in 2 x 1 grid
plot([1999:2015],KMCS_NUM_CH(4,:),'*','MarkerSize',10,'Color','r') 
% plot using + markers
xlabel('Time')
ylabel('Number of clusters'), 
title('Number of clusters due to CalinskiHarabasz criterion')
hold on
plot([1999:2015],FCMS_NUM_CH,'o','MarkerSize',10,'Color','b') 
plot([1999:2015],HCTS_NUM_CH(6,:),'+','MarkerSize',10,'Color',color5) 
plot([1999:2015],CMCS_NUM_CH(7,:),'x','MarkerSize',10,'Color','m') 
hold off

%%%%%%%%%%%%%%%%
figure
subplot(2,1,1)       % add first plot in 2 x 1 grid
plot(Datelong, KSCAPTCoutlong_tv_DB(3,:),'Color','r','LineWidth',lineThick)
%xlabel('Time')
ylabel('Portfolio Wealth')
title('1/n portfolios with time-varying number of clusters')
ax = gca;
ax.XTickLabel = {};
hold on
plot(Datelong,FSCAPTCoutlong_tv_DB,'Color','b', 'LineWidth',lineThick)
plot(Datelong,HSCAPTCoutlong_tv_DB(6,:),'Color',color5,'LineWidth',lineThick)
plot(Datelong,CSCAPTCoutlong_tv_DB(7,:),'Color','m','LineWidth',lineThick)
hold off

subplot(2,1,2)       % add second plot in 2 x 1 grid
plot([1999:2015],KMCS_NUM_DB(3,:),'*','MarkerSize',10,'Color','r') 
% plot using + markers
xlabel('Time')
ylabel('Number of clusters'), 
title('Number of clusters due to DaviesBouldin criterion')
hold on
plot([1999:2015],FCMS_NUM_DB,'o','MarkerSize',10,'Color','b') 
plot([1999:2015],HCTS_NUM_DB(6,:),'+','MarkerSize',10,'Color',color5) 
plot([1999:2015],CMCS_NUM_DB(7,:),'x','MarkerSize',10,'Color','m') 
hold off

%% Plots of one distance vs number of clusters (time series)

% Create color sequence (Rainbow)



% Datelong = [YEAR(2).Date;  YEAR(3).Date;  YEAR(4).Date;  YEAR(5).Date;  YEAR(6).Date;...
%             YEAR(7).Date;  YEAR(8).Date;  YEAR(9).Date;  YEAR(10).Date; YEAR(11).Date;...
%             YEAR(12).Date; YEAR(13).Date; YEAR(14).Date; YEAR(15).Date;...
%             YEAR(16).Date; YEAR(17).Date; YEAR(18).Date]';
        
%Datelong =  PRICE(263:end,1)+693960;
lineThick = 4;

% Max Sharpe ratios 1/n portfolios from KM clusters vs Index (15 TS)

%Figure 1a with TC
cm=colormap(hsv(clus_num-1));
figure
for n = 1:clus_num-1
plot(Datelong, KSCAPTCoutlong (1+(n-1)*length(kmcdist),:),'Color',rainbow14(n,:),'LineWidth',lineThick)
%grid on
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365])
set(gca,'fontsize',18)
 ylim([0 35])
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365*2])
datetick('x','yyyy', 'keepticks')
xlabel('Time')
ylabel('Portfolio Wealth'), title('1/n portfolios from k-means clusters') %select the benchmark here
plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
plot(Datelong, MVCAPTClong (1,1:end),'--','Color','k','LineWidth',lineThick)
plot(Datelong, CAPTClong (1,1:end),'--','Color','k','LineWidth',2)
hold on
end


KSCAPTCtable = [];
for n = 1:clus_num-1
a = KSCAPTCoutlong (1+(n-1)*length(kmcdist),end)
KSCAPTCtable = [KSCAPTCtable;a];
end

KSMVCAPTCtable = [];
for n = 1:clus_num-1
a = KSMVCAPTCoutlong (1+(n-1)*length(kmcdist),end)
KSMVCAPTCtable = [KSMVCAPTCtable;a];
end
% Max Sharpe ratios MV portfolios from KM clusters vs Index (15 TS)

%Figure 1b with TC

figure
for n = 1:clus_num-1
plot(Datelong, KSMVCAPTCoutlong (1+(n-1)*length(kmcdist),:),'Color',rainbow14(n,:),'LineWidth',lineThick)
%grid on
ylim([0 35])
set(gca,'fontsize',18)
datetick('x','yyyy', 'keepticks')

xlabel('Time')
ylabel('Portfolio Wealth'), title('Markowitz portfolios from k-means clusters') %select the benchmark here
plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
plot(Datelong, MVCAPTClong (1,1:end),'--','Color','k','LineWidth',lineThick)
plot(Datelong, CAPTClong (1,1:end),'--','Color','k','LineWidth',2)
hold on
end

% Max Sharpe ratios 1/n portfolios from FCM clusters vs Index (15 TS)

%Figure 2a with TC
cm=colormap(hsv(clus_num-1));
figure
for n = 1:clus_num-1
plot(Datelong, FSCAPTCoutlong (n,:),'Color',rainbow14(n,:),'LineWidth',lineThick)
set(gca,'fontsize',18)
ylim([0 18])
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365])
datetick('x','yyyy', 'keepticks')

xlabel('Time')
ylabel('Portfolio Wealth'), title('1/n portfolios from FUZZY clusters') %select the benchmark here
plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
plot(Datelong, MVCAPTClong (1,1:end),'--','Color','k','LineWidth',lineThick)
plot(Datelong, CAPTClong (1,1:end),'--','Color','k','LineWidth',2)
hold on
end

% Max Sharpe ratios MV portfolios from FCM clusters vs Index (15 TS)

%Figure 2b with TC

figure
for n = 1:clus_num-1
plot(Datelong, FSMVCAPTCoutlong (n,:),'Color',rainbow14(n,:),'LineWidth',lineThick)
%grid on
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365])
datetick('x','yyyy', 'keepticks')
set(gca,'fontsize',18)
ylim([0 18])
xlabel('Time')
ylabel('Portfolio Wealth'), title('Markowitz portfolios from FUZZY clusters') %select the benchmark here
plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
plot(Datelong, MVCAPTClong (1,1:end),'--','Color','k','LineWidth',lineThick)
plot(Datelong, CAPTClong (1,1:end),'--','Color','k','LineWidth',2)
hold on
end

figure
for n = 1:clus_num-1
plot(Datelong, (n+1)*ones(1,length(Datelong)),'Color',rainbow14(n,:),'LineWidth',lineThick)
%grid on
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365])
datetick('x','yyyy', 'keepticks')

xlabel('Time')
ylabel('Number of clusters'), title('Colormap') %select the benchmark here
%plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
hold on
end



% Max Sharpe ratios 1/n portfolios from CM clusters vs Index (15 TS)
%Figure 3a with TC
cm=colormap(hsv(clus_num-1));
figure
for n = 1:clus_num-1
plot(Datelong, CSCAPTCoutlong (1+(n-1)*length(cmcdist),:),'Color',rainbow14(n,:),'LineWidth',lineThick)

%grid on
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365])
hold on
set(gca,'fontsize',18) 
ylim([0 18])
datetick('x','yyyy', 'keepticks')

xlabel('Time')
ylabel('Portfolio Wealth'), title('1/n portfolios from C-medoids clusters') %select the benchmark here

plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
plot(Datelong, MVCAPTClong (1,1:end),'--','Color','k','LineWidth',lineThick)
plot(Datelong, CAPTClong (1,1:end),'--','Color','k','LineWidth',2)


end

% Max Sharpe ratios MV portfolios from CM clusters vs Index (15 TS)
%Figure 3b with TC

figure
for n = 1:clus_num-1
plot(Datelong, CSMVCAPTCoutlong (1+(n-1)*length(cmcdist),:),'Color',rainbow14(n,:),'LineWidth',lineThick)
%grid on
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365])
datetick('x','yyyy', 'keepticks')
set(gca,'fontsize',18)
ylim([0 18])
xlabel('Time')
ylabel('Portfolio Wealth'), title('Markowitz portfolios from C-medoids clusters') %select the benchmark here

plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
plot(Datelong, MVCAPTClong (1,1:end),'--','Color','k','LineWidth',lineThick)
plot(Datelong, CAPTClong (1,1:end),'--','Color','k','LineWidth',2)
hold on
end

CSCAPTCtable = [];
for n = 1:clus_num-1
a = CSCAPTCoutlong (1+(n-1)*length(cmcdist),end)
CSCAPTCtable = [CSCAPTCtable;a];
end

CSMVCAPTCtable = [];
for n = 1:clus_num-1
a = CSMVCAPTCoutlong (1+(n-1)*length(cmcdist),end)
CSMVCAPTCtable = [CSMVCAPTCtable;a];
end
% Max Sharpe ratios 1/n portfolios from HCT clusters vs Index (15 TS)

%Figure 4a with TC
cm=colormap(hsv(clus_num-1));
figure
for n = 1:clus_num-1
plot(Datelong, HSCAPTCoutlong (11+(n-1)*length(hctdist)*length(hctalgo),:),'Color',rainbow14(n,:),'LineWidth',lineThick)% for seuclidean dist weghted algo
%grid on
%grid on
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365])
set(gca,'fontsize',18) 
ylim([0 55])
datetick('x','yyyy', 'keepticks')

xlabel('Time')
ylabel('Portfolio Wealth'), title('1/n portfolios from Hierarchical clusters') %select the benchmark here
plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
plot(Datelong, MVCAPTClong (1,1:end),'--','Color','k','LineWidth',lineThick)
plot(Datelong, CAPTClong (1,1:end),'--','Color','k','LineWidth',2)
hold on
end

% Max Sharpe ratios MV portfolios from HCT clusters vs Index (15 TS)

%Figure 4b with TC

figure
for n = 1:clus_num-1
plot(Datelong, HSMVCAPTCoutlong (11+(n-1)*length(hctdist)*length(hctalgo),:),'Color',rainbow14(n,:),'LineWidth',lineThick)% for seuclidean dist weghted algo
%grid on
%xlim([x2mdate(PRICE(1,1)) Datelong(end,1)+365])
set(gca,'fontsize',18)
ylim([0 55])
datetick('x','yyyy', 'keepticks')
xlabel('Time')
ylabel('Portfolio Wealth'), title('Markowitz portfolios from Hierarchical clusters') %select the benchmark here
plot(Datelong, INDCAPITlong (1,1:end),'Color','k','LineWidth',lineThick)
plot(Datelong, MVCAPTClong (1,1:end),'--','Color','k','LineWidth',lineThick)
plot(Datelong, CAPTClong (1,1:end),'--','Color','k','LineWidth',2)
hold on
end

HSCAPTCtable = [];
for n = 1:clus_num-1
a = HSCAPTCoutlong (11+(n-1)*length(hctdist)*length(hctalgo),end)
HSCAPTCtable = [HSCAPTCtable;a];
end

HSMVCAPTCtable = [];
for n = 1:clus_num-1
a = HSMVCAPTCoutlong (11+(n-1)*length(hctdist)*length(hctalgo),end)
HSMVCAPTCtable = [HSMVCAPTCtable;a];
end


%% Out of sample Mean-variance portfolios from maximum Sharpe ratio stocks from every cluster based on absolute values 
tic
nshift = 0

HSWGT326  = [];
SIGMA326  = [];
SKEWN326  = [];
KURT326   = [];
VAR326    = [];
ES326     = [];
BETA326   = [];
SIGMAAV   = [];
SKEWNAV   = [];
KURTAV    = [];
VARAV     = [];
ESAV      = [];
BETAAV    = [];
IND_all   = [];
for n = 2:length(RET_YEAR)-1
       
Data          = YEAR(n).DATA;
Data_old      = YEAR(n-1).DATA;
Tick          = YEAR(n).TICK;
Tick_old      = YEAR(n-1).TICK;


HSMVCAPoutall = []; 

HSMVCAPRET    = [];
 
indhcts_old   = YEAR(n-1).INDHCT_S(326);


num_dig       = 4;
TargRet       = 0.8;
options       = optimset('Algorithm','active-set','MaxFunEvals',100000);



% Portfolios from hct clusters 
 
for i = 1:size(indhcts_old, 1)*size(indhcts_old, 2)

tick_hcts       = Tick_old(cell2mat(indhcts_old(i)));  
indhcts         = find(ismember(Tick,tick_hcts));

%clear hcap hcapp hcappp  hncap hncapp hncappp hscap hscapp hscappp



HSMVCAPout  = []; 
hscappp   = [];
hscap{1}  = 1; 
hscapp(1) = 1;



    for l = 1:size(Data,1)  
     
       
        XI               = Data_old(:,(cell2mat(indhcts_old(i))));
        w0               = ones(1,size(XI,2))./size(XI,2);
        MeanRet          = mean(XI)';
        Ht               = cov(XI);    
        ub               = ones(length(w0),1);
        lb               = zeros(length(w0),1);
        Aeq              = ones(1,length(w0));
        beq              = 1;
        AA               = -MeanRet';
        bb               = -quantile(MeanRet,TargRet);

        [kswgt,iVaR]      = fmincon(@(w)(sqrt(w*Ht*w')),w0,AA,bb,Aeq,beq,lb,ub,[],options);
        kswgt             = round(kswgt.*(10^num_dig))./(10^num_dig);
        Hswgt{i}          = kswgt;
     
        hscap{l}          = sum(cell2mat(hscap(l)));    
% %portfolio appreciation
        hscap{l+1}        = sum(cell2mat(hscap(l)).*kswgt'.*(1 + Data(l,(indhcts))'));
        hscapp            = hscap{l+1};    
        hscappp            = [hscappp,hscapp];
    end
    

    
%     HSMVCAPout      = [1,hscappp]; 
%     HSMVCAPoutall   = [HSMVCAPoutall; HSMVCAPout]; 
%     HSMVCAPRET      = [HSMVCAPRET; price2ret(HSMVCAPout,[], 'Periodic')]; 
    
 

end

% HSMVCAPRETlong    = [HSMVCAPRETlong, HSMVCAPRET];
% 
HSWGT326  = [HSWGT326;Hswgt];


sigma326 = YEAR(n-1).SIGMA((cell2mat(indhcts_old(i))));
skewn326 = YEAR(n-1).SKEWN((cell2mat(indhcts_old(i))));
kurt326  = YEAR(n-1).KURT((cell2mat(indhcts_old(i))));
var326   = YEAR(n-1).VAR2((cell2mat(indhcts_old(i))));
es326    = YEAR(n-1).ES((cell2mat(indhcts_old(i))));
beta326  = YEAR(n-1).BETA((cell2mat(indhcts_old(i))));


SIGMA326 = [SIGMA326; sigma326];
SKEWN326 = [SKEWN326; skewn326];
KURT326  = [KURT326; kurt326];
VAR326   = [VAR326; var326];
ES326    = [ES326; es326];
BETA326  = [BETA326; beta326];


SIGMAav = [];
SKEWNav = [];
KURTav  = [];
VARav   = [];
ESav    = [];
BETAav  = [];
IND     = [];
for i = 1:length(cell2mat(Hswgt))
    hcts = cell2mat(YEAR(n-1).HCT_S(11));
    ind = find(hcts(:,length(cell2mat(Hswgt))-1)==i);
    
    sigmaav = mean(YEAR(n-1).SIGMA(:,ind));
    skewnav = mean(YEAR(n-1).SKEWN(:,ind));
    kurtav  = mean(YEAR(n-1).KURT(:,ind));
    varav   = mean(YEAR(n-1).VAR2(:,ind));
    esav    = mean(YEAR(n-1).ES(:,ind));
    betaav  = mean(YEAR(n-1).BETA(:,ind));
    
    SIGMAav = [SIGMAav, sigmaav];
    SKEWNav = [SKEWNav, skewnav];
    KURTav  = [KURTav, kurtav];
    VARav   = [VARav, varav];
    ESav    = [ESav, esav];
    BETAav  = [BETAav, betaav];
    IND     = [IND, length(ind)];
end

SIGMAAV = [SIGMAAV; SIGMAav];
SKEWNAV = [SKEWNAV; SKEWNav];
KURTAV  = [KURTAV; KURTav];
VARAV   = [VARAV; VARav];
ESAV    = [ESAV; ESav];
BETAAV  = [BETAAV; BETAav];
IND_all = [IND_all; IND];
nshift  = nshift + 1
end
toc %ca 
HSWGT326  = cell2mat(HSWGT326);

%% Tables with weights (clusters' portraits)
HSWGTTABLE = [];
for m = 2:3%length(RET_YEAR)-1
  HSWGTtable = [];  
  HSWGT = YEAR(m).Hswgt;  
for n = 1:clus_num-1
a = HSWGT(1,11+(n-1)*length(hctdist)*length(hctalgo))
HSWGTtable = [HSWGTtable,a];
end
HSWGTTABLE = [HSWGTTABLE;HSWGTtable];
end


%%%%%%%%%%%%%%%%%%%%%%
%% Clusters 3D vizu
X = YEAR(2).X;
X_std  = bsxfun(@rdivide,X,std(X)); %standartized

C = YEAR(2).HCT{6,7}(:,2);
StD  = X(:,1);
Skew = X(:,2);
Kurt = X(:,3);
VaR  = X(:,4);
ES = X(:,5);
Beta = X(:,6);
S = 30
colormap(rainbow14)
figure
scatter3(VaR, ES, StD, S, C, 'filled')
xlabel('VaR')
ylabel('ES')
zlabel('StD')
set(gca,'xtick',[],'ytick',[],'ztick',[])


%Create a moving 3D plot
OptionZ.FrameRate=15;OptionZ.Duration=5.5;OptionZ.Periodic=true;
CaptureFigVid([-20,10;-110,10;-190,80;-290,10;-380,10], 'Naive HCT',OptionZ)

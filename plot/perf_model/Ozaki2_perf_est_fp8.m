function est_TFLOPs = Ozaki2_perf_est_fp8(type,m,n,k,numMod,fastmode,correct,TOPs,TBs,saveflag,savedir)
%
% est_TFLOPs = Ozaki2_perf_est_fp8(type,m,n,k,numMod,fastmode,correct,TOPs,TBs,saveflag,savedir)
%
% Arguments:
%    type     ... Data type: "S" (single), "D" (double), "C" (complex single), "Z" (complex double)
%    m        ... Number of rows of matrix C
%    n        ... Number of columns of matrix C
%    k        ... Inner dimension
%    numMod   ... Number of moduli
%    fastmode ... true: fast mode, false: accurate mode
%    correct  ... Correction factor accounting for arithmetic overhead in pre-/post-processes
%    TOPs     ... actual FP8 performance [TFLOP/s]
%    TBs      ... actual Bandwidth [TB/s]
%    saveflag ... true: save figures (.fig and .png), false: do not save
%    savedir  ... Directory for saved figures (e.g., "./figs/")
%
% On CPUs (e.g., MONAKA-X),
%    correct = 0;               % O(n^2) is negligible
%    TOPs    = 100:100:1000;
%    TBs     = 0.2:0.2:2;
%
% On GPUs (e.g., B200),
%    correct = 2*numMod;        % O(n^2) is NOT negligible
%    TOPs    = 1000:1000:10000;
%    TBs     = 1:1:10;
%
% For reference, for 0 <= phi <= 4,
%    SGEMM:  5 <= numModuli
%    CGEMM:  5 <= numModuli
%    DGEMM: 12 <= numModuli
%    ZGEMM: 12 <= numModuli
%

%%
arguments (Input)
    type     (1,1) string  {mustBeMember(type,["S","D","C","Z"])}     = "D"
    m        (1,1) double  {mustBePositive,mustBeFinite}              = 16384
    n        (1,1) double  {mustBePositive,mustBeFinite}              = 16384
    k        (1,1) double  {mustBePositive,mustBeFinite}              = 16384
    numMod   (1,1) double  {mustBePositive,mustBeFinite}              = 12
    fastmode (1,1) logical                                            = false
    correct  (1,1) double  {mustBeNonnegative}                        = 2*numMod
    TOPs     (:,1) double  {mustBeVector,mustBePositive,mustBeFinite} = 1000:1000:10000
    TBs      (1,:) double  {mustBeVector,mustBePositive,mustBeFinite} = 1:1:10
    saveflag (1,1) logical                                            = false
    savedir  (1,1) string                                             = ""
end

TOPs = sort(TOPs,'descend');
TBs  = sort(TBs,'ascend');

%% Estimate TFLOP/s
if strcmp(type,"C") || strcmp(type,"Z")
    GEMM_cost = 8*m*n*k;
else
    GEMM_cost = 2*m*n*k;
end

th_Kara = 6;
num_mat = (5*numMod - th_Kara + abs(numMod - th_Kara))./2;
Time_Total = Ozaki2_perf_est_fp8_model(type,fastmode);
Time_Total = subs(Time_Total,'m',m);
Time_Total = subs(Time_Total,'n',n);
Time_Total = subs(Time_Total,'k',k);
Time_Total = subs(Time_Total,'correct',correct);
Time_Total = subs(Time_Total,'TOPs',TOPs);
Time_Total = subs(Time_Total,'TBs',TBs);
Time_Total = subs(Time_Total,'numMod',numMod);
Time_Total = subs(Time_Total,'num_mat',num_mat);
Time_Total = double(Time_Total);
est_TFLOPs = GEMM_cost ./ Time_Total;


%% plot
fig             = figure;
fig.Position(3) = length(TBs)*40;
fig.Position(4) = length(TOPs)*20;

xvalues = string(TBs);
yvalues = string(TOPs);
cdata   = est_TFLOPs;
h = heatmap(xvalues,yvalues,cdata,Colormap=flip(autumn));
h.ColorbarVisible = 'off';
h.CellLabelFormat = '%0.3g';

if fastmode
    name = "FP8-based Ozaki-II fast " + numMod + " moduli";
    figname = type + "_heatmap_perf-est-Ozaki2-fp8-fast-" + numMod + "moduli-n=" + n + "-c=" + correct;
else
    name = "FP8-based Ozaki-II accu. " + numMod + " moduli";
    figname = type + "_heatmap_perf-est-Ozaki2-fp8-accu-" + numMod + "moduli-n=" + n + "-c=" + correct;
end
xlabel("Bandwidth (TB/s)");
ylabel("FP8-GEMM Perf (TFLOP/s)");
title(name);
set(gca,'FontName','Yu Gothic UI Semibold');

if saveflag
    savefig(fig, savedir + figname);
    exportgraphics(fig, savedir + figname + ".png",'Resolution', 600);
end

end

function Time_Total = Ozaki2_perf_est_fp8_model(type,fastmode)

%%
arguments (Input)
    type     (1,1) string  {mustBeMember(type,["S","D","C","Z"])}     = "D"
    fastmode (1,1) logical                                            = true
end

numMod  = sym('numMod');
m       = sym('m');
n       = sym('n');
k       = sym('k');
correct = sym('correct');
TOPs    = sym('TOPs');
TBs     = sym('TBs');

%% Estimate TFLOP/s
if strcmp(type,"C") || strcmp(type,"Z")
    num8f_in  = 3;
    num8f_out = 2;
else
    num8f_in  = 1;
    num8f_out = 1;
end

num_mat = sym('num_mat');

    function byte = bytes(type)
        if strcmp(type,"S"),   byte = 4;  end
        if strcmp(type,"D"),   byte = 8;  end
        if strcmp(type,"C"),   byte = 8;  end
        if strcmp(type,"Z"),   byte = 16; end
        if strcmp(type,"I32"), byte = 4;  end
        if strcmp(type,"I16"), byte = 2;  end
        if strcmp(type,"I8"),  byte = 1;  end
        if strcmp(type,"F32"), byte = 4;  end
        if strcmp(type,"F8"),  byte = 1;  end
    end

%---------------
% 1. Scaling
%---------------
Time_Scaling = 0;
Time_Scaling = Time_Scaling + bytes(type)*m*k ./ TBs;                         % LD find_amax(A)
Time_Scaling = Time_Scaling + bytes(type)*k*n ./ TBs;                         % LD find_amax(B)
if ~fastmode
    Time_Scaling = Time_Scaling + bytes("I16")*m ./ TBs;                      % ST scaling vector for A
    Time_Scaling = Time_Scaling + bytes("I16")*n ./ TBs;                      % ST scaling vector for B
    Time_Scaling = Time_Scaling + num8f_in*bytes("F8")*k*m ./ TBs;            % ST num8f_in*Abar
    Time_Scaling = Time_Scaling + num8f_in*bytes("F8")*k*n ./ TBs;            % ST num8f_in*Bbar
    Time_Scaling = Time_Scaling + num8f_in*2*m*n*k ./ TOPs;                   % Comp num8f_in*f8GEMM
    Time_Scaling = Time_Scaling + num8f_in*bytes("F32")*m*n ./ TBs;           % LD find_amax(C32f)
    Time_Scaling = Time_Scaling + num8f_in*bytes("F32")*m*n ./ TBs;           % LD find_amax(C32f)
    Time_Scaling = Time_Scaling + bytes("I16")*m*k ./ TBs;                    % LD scaling vector for A
    Time_Scaling = Time_Scaling + bytes("I16")*n ./ TBs;                      % LD scaling vector for B
end
Time_Scaling = Time_Scaling + bytes("I16")*m ./ TBs;                          % ST scaling vector for A
Time_Scaling = Time_Scaling + bytes("I16")*n ./ TBs;                          % ST scaling vector for B
Time_Scaling = Time_Scaling + bytes(type)*m*k ./ TBs;                         % LD A for mod(A,pi)
Time_Scaling = Time_Scaling + bytes(type)*k*n ./ TBs;                         % LD B for mod(B,pi)
Time_Scaling = Time_Scaling + num_mat*bytes("F8")*k*m ./ TBs;                 % ST num_mat*A8f
Time_Scaling = Time_Scaling + num_mat*bytes("F8")*k*n ./ TBs;                 % ST num_mat*B8f
Time_Scaling = Time_Scaling + correct*m*k ./ TBs;                             % correction term for A
Time_Scaling = Time_Scaling + correct*k*n ./ TBs;                             % correction term for B

%---------------
% 2. Matrix Multiplication
%---------------
Time_MatMult = num_mat*2*m*n*k ./ TOPs;                                       % Comp num_mat*f8GEMM

%---------------
% 3. Convert F32 to I16
%---------------
Time_Conv32i8i = 0;
Time_Conv32i8i = Time_Conv32i8i + num_mat*bytes("F32")*m*n ./ TBs;            % LD num_mat*C32f
Time_Conv32i8i = Time_Conv32i8i + num8f_out*numMod*bytes("I16")*m*n ./ TBs;   % ST num8i_out*numModuli*C8f
Time_Conv32i8i = Time_Conv32i8i + correct*m*n ./ TBs;                         % correction term for C

%---------------
% 4. Accumulation, Final reduction, and Inverse scaling
%---------------
Time_Final = 0;
Time_Final = Time_Final + num8f_out*numMod*bytes("I16")*m*n ./ TBs;           % LD num8i_out*numModuli*C8i
Time_Final = Time_Final + bytes("I16")*m*n ./ TBs;                            % LD scaling vector for A
Time_Final = Time_Final + bytes("I16")*m*n ./ TBs;                            % LD scaling vector for B
Time_Final = Time_Final + bytes(type)*m*n ./ TBs;                             % ST C
Time_Final = Time_Final + correct*m*n ./ TBs;                                 % correction term for C

%---------------
% Total time
%---------------
t11 = simplify(subs(Time_Scaling,1/TBs,0));
t12 = simplify(subs(Time_Scaling,1/TOPs,0));
t1 = simplify(collect(simplify(collect(collect(simplify(t12),k),m+n)),1/TBs) + t11);
t2 = simplify(simplify(Time_MatMult) + simplify(Time_Conv32i8i));
t3 = collect(simplify(Time_Final),1/TBs);
Time_Total = t1 + t2 + t3;
t1 = simplify(subs(Time_Total,1/TBs,0));
t2 = simplify(collect(collect(collect(collect(subs(Time_Total,1/TOPs,0),numMod),correct),m+n),numMod));
Time_Total = collect(simplify(t1 + t2),1/TBs);

end

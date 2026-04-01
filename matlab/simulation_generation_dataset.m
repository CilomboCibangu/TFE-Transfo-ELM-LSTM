tic
clear
clc

%% ========================================================================
% simulation_generation_dataset.m
% ========================================================================
% Génération automatisée du dataset final brut pour le diagnostic
% d'anomalies d'un transformateur de puissance à huile.
%
% Le script :
%   - pilote le modèle Simulink
%   - parcourt les combinaisons de paramètres de charge, température,
%     facteur de puissance et défauts
%   - simule les 7 classes d'état
%   - exporte les signaux SCADA en fichiers CSV
%   - génère un fichier index.csv et un fichier scenarios.mat
%
% CLASSES :
%   0 = Normal
%   1 = Surcharge
%   2 = Déséquilibre Phase A
%   3 = Déséquilibre Phase B
%   4 = Déséquilibre Phase C
%   5 = Fan fault
%   6 = Pump fault
%
% SORTIES :
%   - 504 fichiers CSV bruts
%   - index.csv
%   - scenarios.mat
%
% APPLICATION :
%   Utilisé dans le cadre du TFE pour générer un dataset physiquement
%   cohérent à partir d'un modèle Simulink de transformateur OFAF.
% ========================================================================

%% Répertoire de sauvegarde
outDir = fullfile(pwd, 'DATASET_TRANSFO_TEST');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% Modèle Simulink
model = 'FUNA_ACTUALISE';
load_system(model)

%% Recherche des blocs nécessaires
Tamb_block  = find_system(model, 'Name', 'env_path');   Tamb_block  = Tamb_block{1};
Kfan_block  = find_system(model, 'Name', 'Kfan');       Kfan_block  = Kfan_block{1};
Kpump_block = find_system(model, 'Name', 'Kpump');      Kpump_block = Kpump_block{1};
load_block  = find_system(model, 'Name', 'Charge3ph');  load_block  = load_block{1};

disp('Blocs trouves ✔')

%% Paramètres globaux
Sn_MVA     = 100;

STOP_TIME  = 60;      % Durée de simulation (s)
EXPORT_TS  = 1e-3;    % Pas d'export des données (s)
TRIM_START = 0.02;    % Suppression du début transitoire (s)

Tamb_grid   = [20 30 40];
cosphi_grid = [0.85 0.95];

S_normal_pu = [0.60 0.75 0.90];
S_over_pu   = [1.10 1.25 1.40];
S_unbal_pu  = [0.60 0.70 0.80];
S_cool_pu   = [0.60 0.70 0.80];

unbal_pct   = [0.20 0.35];

Kfan_part   = [1.50 1.90];
Kpump_part  = [2.00 2.40];

% Répétitions pour obtenir 72 cas par classe
REP_01  = 4;   % 3*2*3*4 = 72
REP_234 = 2;   % 3*2*3*2*2 = 72
REP_56  = 2;   % 3*2*3*2*2 = 72

%% Structure de stockage des scénarios
results = struct( ...
    'Mode', {}, ...
    'Label', {}, ...
    'S_pu', {}, ...
    'S_MVA', {}, ...
    'Tamb', {}, ...
    'cosphi', {}, ...
    'Fault', {}, ...
    'Kpump', {}, ...
    'Kfan', {}, ...
    'unbal_pct', {}, ...
    'PhaseMono', {}, ...
    'csv', {});

scenario_id = 1;

%% ========================================================================
%% CLASSE 0 — NORMAL
%% ========================================================================
fprintf('\n=== CLASSE 0 : NORMAL ===\n')
set_param(Kfan_block,  'Value', '1')
set_param(Kpump_block, 'Value', '1')

for rep = 1:REP_01
for Tamb = Tamb_grid
for cosphi = cosphi_grid
for S_pu = S_normal_pu

    set_param(Tamb_block, 'Value', num2str(Tamb))

    [P_br, Q_br] = chargeBalancee(S_pu, Sn_MVA, cosphi);
    appliquerCharge(load_block, P_br, Q_br)

    tag = sprintf('CL0_NORMAL_S%03d_T%02d_c%.2f_rep%d', ...
        round(100*S_pu), Tamb, cosphi, rep);

    filePath = simExporterBrut(model, STOP_TIME, EXPORT_TS, TRIM_START, outDir, tag);

    results(scenario_id) = makeRow('NORMAL', 0, S_pu, Sn_MVA, Tamb, cosphi, ...
        'NA', 1.0, 1.0, 0, '-', filePath);

    fprintf('sim %d | Cl0 | S=%d%% Tamb=%d cosphi=%.2f rep=%d\n', ...
        scenario_id, round(S_pu*100), Tamb, cosphi, rep)

    scenario_id = scenario_id + 1;

end
end
end
end

%% ========================================================================
%% CLASSE 1 — SURCHARGE
%% ========================================================================
fprintf('\n=== CLASSE 1 : SURCHARGE ===\n')
set_param(Kfan_block,  'Value', '1')
set_param(Kpump_block, 'Value', '1')

for rep = 1:REP_01
for Tamb = Tamb_grid
for cosphi = cosphi_grid
for S_pu = S_over_pu

    set_param(Tamb_block, 'Value', num2str(Tamb))

    [P_br, Q_br] = chargeBalancee(S_pu, Sn_MVA, cosphi);
    appliquerCharge(load_block, P_br, Q_br)

    tag = sprintf('CL1_OVERLOAD_S%03d_T%02d_c%.2f_rep%d', ...
        round(100*S_pu), Tamb, cosphi, rep);

    filePath = simExporterBrut(model, STOP_TIME, EXPORT_TS, TRIM_START, outDir, tag);

    results(scenario_id) = makeRow('OVERLOAD', 1, S_pu, Sn_MVA, Tamb, cosphi, ...
        'NA', 1.0, 1.0, 0, '-', filePath);

    fprintf('sim %d | Cl1 | S=%d%% Tamb=%d cosphi=%.2f rep=%d\n', ...
        scenario_id, round(S_pu*100), Tamb, cosphi, rep)

    scenario_id = scenario_id + 1;

end
end
end
end

%% ========================================================================
%% CLASSES 2, 3, 4 — DÉSÉQUILIBRE PAR PHASE
%% ========================================================================
desConfig = struct( ...
    'label', {2,      3,      4     }, ...
    'plus',  {[1 3],  [1 2],  [2 3] }, ...
    'minus', {2,      3,      1     }, ...
    'nom',   {'Ph-A', 'Ph-B', 'Ph-C'});

for dc = 1:3
    lbl = desConfig(dc).label;
    ipl = desConfig(dc).plus;
    imi = desConfig(dc).minus;
    nom = desConfig(dc).nom;

    fprintf('\n=== CLASSE %d : DESEQUILIBRE %s ===\n', lbl, nom)
    set_param(Kfan_block,  'Value', '1')
    set_param(Kpump_block, 'Value', '1')

    for rep = 1:REP_234
    for Tamb = Tamb_grid
    for cosphi = cosphi_grid
    for S_pu = S_unbal_pu
    for upct = unbal_pct

        set_param(Tamb_block, 'Value', num2str(Tamb))

        S_VA  = S_pu * Sn_MVA * 1e6;
        Pbase = S_VA * cosphi / 3;
        Qbase = S_VA * sqrt(max(0, 1-cosphi^2)) / 3;

        P_br = [Pbase Pbase Pbase];
        Q_br = [Qbase Qbase Qbase];

        P_br(ipl) = P_br(ipl) + upct * Pbase;
        Q_br(ipl) = Q_br(ipl) + upct * Qbase;
        P_br(imi) = P_br(imi) - 2 * upct * Pbase;
        Q_br(imi) = Q_br(imi) - 2 * upct * Qbase;

        if any(P_br < 0)
            continue
        end

        appliquerCharge(load_block, P_br, Q_br)

        tag = sprintf('CL%d_UNB_%s_S%03d_up%02d_T%02d_c%.2f_rep%d', ...
            lbl, nom, round(100*S_pu), round(100*upct), Tamb, cosphi, rep);

        filePath = simExporterBrut(model, STOP_TIME, EXPORT_TS, TRIM_START, outDir, tag);

        results(scenario_id) = makeRow('UNBALANCE', lbl, S_pu, Sn_MVA, Tamb, cosphi, ...
            'NA', 1.0, 1.0, upct, nom, filePath);

        fprintf('sim %d | Cl%d %s | S=%d%% upct=%d%% Tamb=%d cos=%.2f rep=%d\n', ...
            scenario_id, lbl, nom, round(S_pu*100), round(upct*100), Tamb, cosphi, rep)

        scenario_id = scenario_id + 1;

    end
    end
    end
    end
    end
end

%% ========================================================================
%% CLASSE 5 — FAN FAIL
%% ========================================================================
fprintf('\n=== CLASSE 5 : FAN FAIL ===\n')
set_param(Kpump_block, 'Value', '1')

for rep = 1:REP_56
for Tamb = Tamb_grid
for cosphi = cosphi_grid
for S_pu = S_cool_pu
for Kfan = Kfan_part

    set_param(Tamb_block, 'Value', num2str(Tamb))
    set_param(Kfan_block,  'Value', num2str(Kfan))

    [P_br, Q_br] = chargeBalancee(S_pu, Sn_MVA, cosphi);
    appliquerCharge(load_block, P_br, Q_br)

    tag = sprintf('CL5_FANFAIL_S%03d_Kf%.2f_T%02d_c%.2f_rep%d', ...
        round(100*S_pu), Kfan, Tamb, cosphi, rep);

    filePath = simExporterBrut(model, STOP_TIME, EXPORT_TS, TRIM_START, outDir, tag);

    results(scenario_id) = makeRow('FAN_FAIL', 5, S_pu, Sn_MVA, Tamb, cosphi, ...
        'FAN_FAIL', 1.0, Kfan, 0, '-', filePath);

    fprintf('sim %d | Cl5 | S=%d%% Kfan=%.2f Tamb=%d cos=%.2f rep=%d\n', ...
        scenario_id, round(S_pu*100), Kfan, Tamb, cosphi, rep)

    scenario_id = scenario_id + 1;

end
end
end
end
end

%% ========================================================================
%% CLASSE 6 — PUMP FAIL
%% ========================================================================
fprintf('\n=== CLASSE 6 : PUMP FAIL ===\n')
set_param(Kfan_block, 'Value', '1')

for rep = 1:REP_56
for Tamb = Tamb_grid
for cosphi = cosphi_grid
for S_pu = S_cool_pu
for Kpump = Kpump_part

    set_param(Tamb_block,  'Value', num2str(Tamb))
    set_param(Kpump_block, 'Value', num2str(Kpump))

    [P_br, Q_br] = chargeBalancee(S_pu, Sn_MVA, cosphi);
    appliquerCharge(load_block, P_br, Q_br)

    tag = sprintf('CL6_PUMPFAIL_S%03d_Kp%.2f_T%02d_c%.2f_rep%d', ...
        round(100*S_pu), Kpump, Tamb, cosphi, rep);

    filePath = simExporterBrut(model, STOP_TIME, EXPORT_TS, TRIM_START, outDir, tag);

    results(scenario_id) = makeRow('PUMP_FAIL', 6, S_pu, Sn_MVA, Tamb, cosphi, ...
        'PUMP_FAIL', Kpump, 1.0, 0, '-', filePath);

    fprintf('sim %d | Cl6 | S=%d%% Kpump=%.2f Tamb=%d cos=%.2f rep=%d\n', ...
        scenario_id, round(S_pu*100), Kpump, Tamb, cosphi, rep)

    scenario_id = scenario_id + 1;

end
end
end
end
end

%% ========================================================================
%% SAUVEGARDE FINALE
%% ========================================================================
finaliser(results, outDir)

fprintf('\n========================================\n')
fprintf('TOTAL simulations : %d\n', numel(results))
fprintf('Distribution par classe :\n')
for c = 0:6
    fprintf('  Classe %d : %d simulations\n', c, sum([results.Label] == c))
end

fprintf('\n✔ Dataset final termine dans : %s\n', outDir)
fprintf('  - 504 CSV bruts\n')
fprintf('  - index.csv\n')
fprintf('  - scenarios.mat\n')

toc

%% ========================================================================
%% FONCTIONS LOCALES
%% ========================================================================

function row = makeRow(modeName, label, S_pu, Sn_MVA, Tamb, cosphi, faultLabel, Kpump, Kfan, upct, phaseMono, filePath)
row = struct( ...
    'Mode', modeName, ...
    'Label', label, ...
    'S_pu', S_pu, ...
    'S_MVA', S_pu * Sn_MVA, ...
    'Tamb', Tamb, ...
    'cosphi', cosphi, ...
    'Fault', faultLabel, ...
    'Kpump', Kpump, ...
    'Kfan', Kfan, ...
    'unbal_pct', upct, ...
    'PhaseMono', phaseMono, ...
    'csv', filePath);
end

function [P_br, Q_br] = chargeBalancee(S_pu, Sn_MVA, cosphi)
    S_VA  = S_pu * Sn_MVA * 1e6;
    P_tot = S_VA * cosphi;
    Q_tot = S_VA * sqrt(max(0, 1 - cosphi^2));
    P_br  = [P_tot/3 P_tot/3 P_tot/3];
    Q_br  = [Q_tot/3 Q_tot/3 Q_tot/3];
end

function appliquerCharge(load_block, P_br, Q_br)
    P_br = P_br(:).';
    Q_br = Q_br(:).';

    try, set_param(load_block, 'UnbalancedPower', 'on'); catch, end
    try, set_param(load_block, 'UnbalancedPower', '1');  catch, end

    set_param(load_block, 'Pabcp',  mat2str(P_br))
    set_param(load_block, 'QLabcp', mat2str(Q_br))
    set_param(load_block, 'QCabcp', mat2str([0 0 0]))

    try, set_param(load_block, 'Pabc',  mat2str(P_br)); catch, end
    try, set_param(load_block, 'QLabc', mat2str(Q_br)); catch, end
    try, set_param(load_block, 'QCabc', mat2str([0 0 0])); catch, end
end

function filePath = simExporterBrut(model, STOP_TIME, EXPORT_TS, TRIM_START, outDir, tag)

    evalin('base', "if exist('SCADA_DATA','var'), clear SCADA_DATA; end");

    simIn = Simulink.SimulationInput(model);
    simIn = simIn.setModelParameter( ...
        'StopTime', num2str(STOP_TIME), ...
        'ReturnWorkspaceOutputs', 'on', ...
        'SolverType', 'Variable-step', ...
        'Solver', 'ode23t', ...
        'MaxStep', '5e-5');

    simOut = sim(simIn);

    SC = [];
    try
        SC = simOut.get('SCADA_DATA');
    catch
    end

    if isempty(SC) && evalin('base', "exist('SCADA_DATA','var')")
        SC = evalin('base', 'SCADA_DATA');
    end

    if isempty(SC)
        error("SCADA_DATA introuvable. Verifie le bloc To Workspace (SCADA_DATA).");
    end

    if ~isa(SC, 'timeseries')
        error("SCADA_DATA doit etre un timeseries.");
    end

    tEnd = SC.Time(end);
    tq = (0:EXPORT_TS:tEnd).';
    SCr = resample(SC, tq);

    t = SCr.Time(:);
    X = SCr.Data;

    n = min(numel(t), size(X, 1));
    t = t(1:n);
    X = X(1:n, :);

    if TRIM_START > 0
        keep = t >= TRIM_START;
        t = t(keep);
        X = X(keep, :);
    end

    if size(X, 2) < 17
        error("SCADA_DATA.Data a %d colonnes. Attendu >=17.", size(X, 2));
    end

    V_HT  = X(:, 1:3);
    I_HT  = X(:, 4:6);
    V_MT  = X(:, 7:9);
    I_MT  = X(:, 10:12);
    T_oil = X(:, 13);
    T_wdg = X(:, 14:16);
    T_amb = X(:, 17);

    TT = timetable(seconds(t), ...
        V_HT(:,1), V_HT(:,2), V_HT(:,3), ...
        I_HT(:,1), I_HT(:,2), I_HT(:,3), ...
        V_MT(:,1), V_MT(:,2), V_MT(:,3), ...
        I_MT(:,1), I_MT(:,2), I_MT(:,3), ...
        T_oil, T_wdg(:,1), T_wdg(:,2), T_wdg(:,3), T_amb, ...
        'VariableNames', {'V_HTa','V_HTb','V_HTc', ...
                          'I_HTa','I_HTb','I_HTc', ...
                          'V_MTa','V_MTb','V_MTc', ...
                          'I_MTa','I_MTb','I_MTc', ...
                          'T_oil','T_wdg_A','T_wdg_B','T_wdg_C','T_amb'});

    T = timetable2table(TT, 'ConvertRowTimes', true);
    T.Properties.VariableNames{1} = 'Temps';
    T.Temps = seconds(T.Temps);

    vn = T.Properties.VariableNames;
    for ii = 1:numel(vn)
        if isnumeric(T.(vn{ii}))
            T.(vn{ii}) = round(T.(vn{ii}), 6);
        end
    end

    fileName = ['sim_' tag '.csv'];
    fileName = regexprep(fileName, '(?<=\d)\.(?=\d)', 'p');

    filePath = fullfile(outDir, fileName);
    writetable(T, filePath, 'Delimiter', ';');
end

function finaliser(results, outDir)
    Tindex = struct2table(results);
    writetable(Tindex, fullfile(outDir, 'index.csv'), 'Delimiter', ';');
    save(fullfile(outDir, 'scenarios.mat'), 'results');
end

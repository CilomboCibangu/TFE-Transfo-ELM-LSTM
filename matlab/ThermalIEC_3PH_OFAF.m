function [T_oil, T_wdg_abc] = ThermalIEC_3PH_OFAF(Irms_abc, Tamb, Kpump, Kfan, reset)
% =========================================================================
% ThermalIEC_3PH_OFAF
% =========================================================================
% Modèle thermique différentiel triphasé inspiré de l'IEC 60076-7 pour un
% transformateur de puissance à huile en régime OFAF
% (Oil Forced Air Forced).
%
% Ce modèle calcule à chaque pas de temps :
%   - la température d'huile supérieure (top-oil)
%   - la température des enroulements des phases A, B et C
%
% ENTREES :
%   Irms_abc  [1x3]  Courants RMS des phases A, B, C (A)
%   Tamb      [1x1]  Température ambiante (°C)
%   Kpump     [1x1]  Facteur de dégradation de la pompe
%                    1 = normal, >1 = dégradé
%   Kfan      [1x1]  Facteur de dégradation des ventilateurs
%                    1 = normal, >1 = dégradé
%   reset     [1x1]  Réinitialisation des états persistants
%                    0 = conserver l'état précédent
%                    1 = réinitialiser
%
% SORTIES :
%   T_oil      [1x1]  Température d'huile supérieure (°C)
%   T_wdg_abc  [1x3]  Températures des enroulements A, B, C (°C)
%
% HYPOTHESES DU MODELE :
%   - Les pertes cuivre varient avec le carré du courant
%   - Kfan agit principalement sur l'échange huile <-> ambiance
%   - Kpump agit principalement sur l'échange enroulement <-> huile
%   - La dynamique est intégrée par Euler explicite
%
% APPLICATION :
%   Utilisé pour la génération d'un dataset de signatures thermiques
%   physiquement cohérentes dans le cadre du TFE sur la détection et la
%   classification d'anomalies dans un transformateur à huile.
% =========================================================================

%% Valeur par défaut pour reset
if nargin < 5
    reset = 0;
end

%% Pas de temps
% Valide uniquement si la fonction est appelée toutes les 0.01 s
Ts = 0.01;

%% Mise en forme des entrées
Irms_abc = reshape(double(Irms_abc), 1, 3);
Tamb     = double(Tamb);
Kpump    = max(1, double(Kpump));
Kfan     = max(1, double(Kfan));

%% Paramètres nominaux du transformateur
Sn     = 100e6;     % Puissance apparente nominale (VA)
Vll_MT = 20e3;      % Tension ligne-ligne côté MT (V)
Irated = Sn / (sqrt(3) * Vll_MT);

Pfe = 120e3;        % Pertes fer (W)
Pcu = 450e3;        % Pertes cuivre à charge nominale (W)
R   = Pcu / Pfe;    % Rapport pertes cuivre / pertes fer

%% Paramètres thermiques OFAF
n_exp = 1.0;
m_exp = 1.0;

dTo_rated_nom = 45;      % Elévation nominale top-oil (°C)
dTw_rated_nom = 30;      % Gradient nominal enroulement-huile (°C)

tau_oil_nom = 90 * 60;   % Constante de temps huile (s)
tau_wdg_nom = 7 * 60;    % Constante de temps enroulement (s)

%% Effets des défauts de refroidissement
% Kfan  -> agit surtout sur huile <-> ambiance
% Kpump -> agit surtout sur enroulement <-> huile
dTo_rated = dTo_rated_nom * Kfan;
tau_oil   = tau_oil_nom   * Kfan;

dTw_rated = dTw_rated_nom * Kpump;
tau_wdg   = tau_wdg_nom   * Kpump;

%% Charge et pertes
K       = Irms_abc / Irated;      % Facteur de charge par phase
Pcu_ph  = (Pcu / 3) * (K .^ 2);   % Pertes cuivre par phase
Pcu_tot = sum(Pcu_ph);            % Pertes cuivre totales
K2_eq   = max(0, Pcu_tot / Pcu);  % Facteur équivalent de charge au carré

%% Régime permanent
% Elévation top-oil
dTo_ult = dTo_rated * ((K2_eq * R + 1) / (R + 1)) ^ n_exp;

% Gradient enroulement-huile par phase
dTw_ult = dTw_rated * (max(0, K) .^ 2) .^ m_exp;

% Température finale d'huile
To_ult = Tamb + dTo_ult;

%% Etats persistants
persistent To dTw

if isempty(To) || reset == 1
    To  = Tamb + 20;   % Huile initiale
    dTw = [0 0 0];     % Gradient initial
end

%% Dynamique thermique
% Huile
To = To + Ts * ((To_ult - To) / tau_oil);

% Gradient enroulement-huile
dTw = dTw + Ts * ((dTw_ult - dTw) / tau_wdg);

% Température des enroulements
Tw = To + dTw;

%% Sorties
T_oil     = To;
T_wdg_abc = Tw;

end

% Walking Robot Startup Script
%
% Copyright 2017-2019 The MathWorks, Inc.

%% Clear everything
clc
clear
close all

%% Add folders to the path
addpath(genpath('ReinforcementLearning'));  % Reinforcement learning files
        
%% Load basic robot parameters from modeling and simulation example
robotParametersRL

%% Open the README file
%edit README.md
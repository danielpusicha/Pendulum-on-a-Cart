clear
clc

linearizedController

mdl = 'mySimRL';
open_system(mdl)

% Define the observation specification obsInfo and action specification actInfo.
obsFeat = 4;
obsInfo = rlNumericSpec([obsFeat 1]);
obsInfo.Name = 'observations';
obsInfo.Description = 'position x, time derivative position x dot, deflection theta, time derivative deflection theta dot';
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1]);
actInfo.Name = 'force';
numActions = actInfo.Dimension(1);

% Build the environment interface object.
env = rlSimulinkEnv('mySimRL','mySimRL/RL Agent',...
        obsInfo,actInfo);
    
% Specify the simulation time Tf and the agent sample time Ts in seconds.  
Ts = 0.1;
Tf = 30;

% Fix the random generator seed for reproducibility.
rng(0)

% Specify the quadratic cost function
Q = eye(4).*[1; 1; 100; 50];
R = 1;
%% Create DDPG Agent

% Given observations and actions, a DDPG agent approximates the long-term
% reward using a critic value function representation.
%To create the critic, first create a deep neural network with two inputs,
% the observation and action, and one output.
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(50,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(25,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','Action')
    fullyConnectedLayer(25,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

% View the critic network configuration.
figure
plot(criticNetwork)

% Specify options for the critic representation using rlRepresentationOptions.
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);

% Create the critic representation using the specified deep neural network and options.
% You must also specify the action and observation specifications for the critic,
% which you obtain from the environment interface.
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);

% Given observations, a DDPG agent decides which action to take using an actor representation.
% To create the actor, first create a deep neural network with one input,
% the observation, and one output, the action.

% Construct the actor in a similar manner to the critic. 
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(3, 'Name','actorFC')
    tanhLayer('Name','actorTanh')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
% To create the DDPG agent, first specify the DDPG agent options using rlDDPGAgentOptions.
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1.0, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agentOpts.NoiseOptions.Variance = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;

% Then, create the DDPG agent using the specified actor representation, critic representation, and agent options.
% Set to true, to resume training from a saved agent
resumeTraining = false;
% Set ResetExperienceBufferBeforeTraining to false to keep experience from the previous session
%%agentOpts.ResetExperienceBufferBeforeTraining = ~(resumeTraining);
if resumeTraining
    % Load the agent from the previous session
    sprintf('- Resume training of: %s', 'SwingUpDDPG.mat');   
    load('SwingUpDDPG.mat','agent');
else
    % Create a fresh new agent
    agent = rlDDPGAgent(actor, critic, agentOpts);
end

%% Train Agent

% To train the agent, first specify the training options. For this example, use the following options:

% *Run each training for at most 5000 episodes. 
%  Specify that each episode lasts for at most ceil(Tf/Ts) (that is 300) time steps.
% *Display the training progress in the Episode Manager dialog box (set the Plots option)
%  and disable the command line display (set the Verbose option to false).
% *Stop training when the agent receives an average cumulative reward
%  greater than 10 over 20 consecutive episodes. At this point, 
%  the agent can control the swing up process.
maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
% if exist ./'Pendulum on Cart Repository'/'RL Agents' == 0
%     mkdir ./'Pendulum on Cart Repository'/'RL Agents'
% end
agentdir = pwd + "/Pendulum on Cart Repository" + "/RL Agents";
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',20, ...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',8,...
    'SaveAgentDirectory', 'Pendulum on Cart Repository/RL Agents/SwingUpDDPG.mat',...
    'Verbose',true, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',1.0000e+20);

% % Training options for parallelization
% trainOpts.UseParallel = true;
% trainOpts.ParallelizationOptions.Mode = "async";
% trainOpts.ParallelizationOptions.DataToSendFromWorkers = "experiences";
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;

% Train the agent using the train function. Training is a computationally intensive process 
% that takes several minutes to complete. To save time while running this example, 
% load a pretrained agent by setting doTraining to false. To train the agent yourself, 
% set doTraining to true.
doTraining = false;

if doTraining
    % set the status of the Test Switch: 1 -> RL Modus
    set_param('mySimRL/Test Switch','sw','1')
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    %save ('SwingUpDDPG.mat','agent')
    save ('SwingUpDDPG.mat','agent')
    % currentfig = findall(groot,'Type','Figure');
    % savefig(currentfig,'training.fig')
else
    % Load the pretrained agent for the example.
    %load('SwingUpDDPG.mat','agent')
    load('SwingUpDDPG.mat','saved_agent')
end
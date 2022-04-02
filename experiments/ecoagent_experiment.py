# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run eco-agent experiment."""

import json
import os, shutil

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
import training_utils

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from recs_ecosystem_creator_rl.environment import environment
from recs_ecosystem_creator_rl.recommender import agent
from recs_ecosystem_creator_rl.recommender import data_utils
from recs_ecosystem_creator_rl.recommender import runner
from recs_ecosystem_creator_rl.recommender import value_model
from sklearn import preprocessing
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 7e-4, 'Learning rate.')
# User value model configs.
flags.DEFINE_string('user_rnn_type', 'LSTM', 'User recurrent cell.')
flags.DEFINE_string('user_hidden_sizes', '32,32,16',
                    'Sizes of hidden layers to embed user history.')

# Creator value model configs.
flags.DEFINE_string('creator_rnn_type', 'LSTM', 'Creator recurrent cell.')
flags.DEFINE_string('creator_hidden_sizes', '32,32,16',
                    'Sizes of hidden layers to embed creator history.')
flags.DEFINE_integer('creator_id_embedding_size', 32,
                     'Size of creator id embedding.')

# Actor model configs.
flags.DEFINE_string('actor_hidden_sizes', '32,32,32',
                    'Sizes of hidden layers in actor model.')
flags.DEFINE_integer('actor_weight_size', 16,
                     'Size of weights for softmax in actor model.')
flags.DEFINE_float('actor_learning_rate', 7e-4, 'Learning rate in actor model.')
flags.DEFINE_float('actor_entropy_coeff', 0,
                   'Entropy coefficient in loss function in actor model.')
flags.DEFINE_float('social_reward_coeff', 0.0,
                   'Coefficient of social reward in actor model optimization.')
flags.DEFINE_float(
    'loss_denom_decay', 0.0,
    'Momentum for moving average of label weights normalization.')

# Environment configs.
flags.DEFINE_float('large_recommendation_reward', 2.0,
                   'Large recommendation reward for creators.')
flags.DEFINE_float('small_recommendation_reward', 0.5,
                   'Small recommendation reward for creators.')
flags.DEFINE_string(
    'copy_varied_property', None,
    'If not None, generate two identical creator groups but vary the specified property.'
)

# Runner configs.
flags.DEFINE_integer('nsteps', 1600, 'Maximum length of a trajectory.')
flags.DEFINE_float('user_gamma', 0.99, 'Discount factor for user utility.')
flags.DEFINE_float('creator_gamma', 0.99,
                   'Discount factor for creator utility.')

# Training configs.
flags.DEFINE_float('random_user_accumulated_reward', 42.4,
                   'Average user accumulated reward from random agent.')
flags.DEFINE_float('random_creator_accumulated_reward', 3.9,
                   'Average creator accumulated reward from random agent.')
flags.DEFINE_string('logdir', '', 'Log directory.')
flags.DEFINE_integer('epochs', 300,
                     'The number of epochs to run training for.')
flags.DEFINE_integer('start_save', 25,
                     'The number of epochs to run before saving models.')
flags.DEFINE_integer('save_frequency', 25, 'Frequency of saving.')
flags.DEFINE_integer('summary_frequency', 25, 'Frequency of writing summaries.')
flags.DEFINE_integer('epoch_runs', 5, 'The number of runs to collect data.')
flags.DEFINE_integer('epoch_trains', 1, 'The number of trains per epoch.')
flags.DEFINE_integer('batch_size', 32,
                     'The number of trajectories per training batch.')


def learn(env_config, user_value_model_config, creator_value_model_config,
          actor_model_config, exp_config):
  """Train and test user_value_model and creator_value_model with random agent."""

  # Random agent normalization.
  random_user_accumulated_reward = FLAGS.random_user_accumulated_reward
  random_creator_accumulated_reward = FLAGS.random_creator_accumulated_reward

  env = environment.create_gym_environment(env_config)
  user_value_model = value_model.UserValueModel(**user_value_model_config)
  creator_value_model = value_model.CreatorValueModel(
      **creator_value_model_config)
  actor_model = agent.PolicyGradientAgent(
      user_model=user_value_model,
      creator_model=creator_value_model,
      **actor_model_config)

  runner_ = runner.Runner(env, actor_model, exp_config['nsteps'])

  experience_replay = data_utils.ExperienceReplay(exp_config['nsteps'],
                                                  env_config['topic_dim'],
                                                  env_config['num_candidates'],
                                                  exp_config['user_gamma'],
                                                  exp_config['creator_gamma'])

  train_summary_dir = os.path.join(FLAGS.logdir, 'train/')
  #os.makedirs(train_summary_dir)
  if( not os.path.isdir(train_summary_dir) ):
      #shutil.rmtree(train_summary_dir)
      os.makedirs(train_summary_dir)

  train_summary_writer = tf.summary.create_file_writer(train_summary_dir)
  fig = plt.figure(figsize=(30,15))
  # Train, save.

 # For each epoch, EcoAgent interacts with 10 new environments as set up above. The environment will be rolled out for 20 steps. At each time step of one rollout, all users receive recommendations simultaneously, and the environment updates all users’ and content providers’ states.
  for epoch in range(exp_config['epochs']):
    print("epochs", epoch)

    num_users = []  # Shape (sum(run_trajectory_length)).
    num_creators = []  # Shape (sum(run_trajectory_length)).
    num_documents = []  # Shape (sum(run_trajectory_length)).

    topic_distribution = []
    topic_sum = []

    selected_probs = []  # Shape (sum(run_trajectory_length)).
    policy_probs = []  # Shape (sum(run_trajectory_length), num_candidates).
    # Collect training data.
    for _ in range(exp_config['epoch_runs']):                   #simulation
      (user_dict, creator_dict, preprocessed_user_candidates, _, probs, _,
       _, topic_distribution, topic_sum) = runner_.run()
      experience_replay.update_experience(
          user_dict,
          creator_dict,
          preprocessed_user_candidates,
          update_actor=True)
      num_users.append(runner_.env.num_users)

      num_creators.append(runner_.env.num_creators)      #num viable creator at the end of the simulation
      num_documents.append(runner_.env.num_documents)
      selected_probs.extend(probs['selected_probs'])
      policy_probs.extend(probs['policy_probs'])
     # print(num_users, num_creators)
      #topic_distribution.append(runner_.env.topic_documents)  #topic distribution at the end of the experiment, I want to analyse the document distribution at each step of the simulation

    #print(np.array(topic_distribution).shape)

    # Update user and content creators state model with training data.
    for _ in range(exp_config['epoch_trains']):
      for (inputs, label, user_utility, social_reward,
           _) in experience_replay.actor_data_generator(
               creator_value_model, batch_size=exp_config['batch_size']):
        actor_model.train_step(
            inputs, label, user_utility / random_user_accumulated_reward,
            social_reward / random_creator_accumulated_reward)

      for batch_data in experience_replay.user_data_generator(
          exp_config['batch_size']):
        user_value_model.train_step(*batch_data)
      for batch_data in experience_replay.creator_data_generator(
          exp_config['batch_size'],
          creator_id_embedding_size=creator_value_model
          .creator_id_embedding_size):
        creator_value_model.train_step(*batch_data)

    sum_user_normalized_accumulated_reward = np.sum(
        list(experience_replay.user_accumulated_reward.values())
    ) / experience_replay.num_runs / random_user_accumulated_reward

    sum_creator_normalized_accumulated_reward = np.sum(
        list(experience_replay.creator_accumulated_reward.values())
    ) / experience_replay.num_runs / random_creator_accumulated_reward

    overall_scaled_accumulated_reward = (
        (1 - actor_model_config['social_reward_coeff']) *
        sum_user_normalized_accumulated_reward +
        actor_model_config['social_reward_coeff'] *
        sum_creator_normalized_accumulated_reward)
    # Write summary statistics for tensorboard.
    if epoch % exp_config['summary_frequency'] == 0:
      ## Value model and environment summaries.
      training_utils.save_summary(train_summary_writer, user_value_model,
                                  creator_value_model, experience_replay,
                                  num_users, num_creators, num_documents, topic_distribution,
                                  policy_probs, selected_probs,
                                  overall_scaled_accumulated_reward, epoch)
      ## Actor model summaries.
      with train_summary_writer.as_default():
        tf.summary.scalar(
            'actor_loss', actor_model.train_loss.result(), step=epoch)

        social_rewards = np.array(
            experience_replay.actor_creator_uplift_utilities)
        tf.summary.scalar('social_rewards', np.mean(social_rewards), step=epoch)
        actor_label_weights = (
            (1 - actor_model_config['social_reward_coeff']) *
            np.array(experience_replay.actor_user_utilities) /
            random_user_accumulated_reward +
            actor_model_config['social_reward_coeff'] * social_rewards /
            random_creator_accumulated_reward)
        tf.summary.scalar(
            'actor_label_weights', np.mean(actor_label_weights), step=epoch)
        #document distribution for tensorboard
        """
            with train_summary_writer.as_default():
                for step in range(20):
                for topic in range(10):
                    st = 'topic_distribution epoch  ' + str(epoch)
                    tf.summary.scalar(
                        st , np.array(topic_distribution)[step,topic], step=step)"""

    # Reset.
    user_value_model.train_loss.reset_states()
    user_value_model.train_relative_loss.reset_states()
    creator_value_model.train_loss.reset_states()
    creator_value_model.train_reEcoAlative_loss.reset_states()
    actor_model.train_loss.reset_states()
    actor_model.train_utility_loss.reset_states()
    actor_model.train_entropy_loss.reset_states()
    experience_replay.reset()
    if(epoch == 0):
      ax = fig.add_subplot(3,5, 1)

      x = np.arange(20)
      for topic in range(10):
        ax.plot(x, np.array(topic_distribution)[:,topic], label = str(topic))
        ax.set_title("epoch " + str(epoch))
        ax.set_ylabel("distribution")
        ax.set_xlabel("interaction step")
        ax.legend()


    if epoch >= exp_config['start_save'] and exp_config[
        'save_frequency'] > 0 and epoch % exp_config['save_frequency'] == 0:
      # Save model.
      user_value_model.save()
      creator_value_model.save()
      actor_model.save()
      #visualization of topic distribution over time
      ax = fig.add_subplot(3,5, int(epoch/25)+1)

      x = np.arange(20)
      for topic in range(10):
        ax.plot(x, np.array(topic_distribution)[:,topic], label = str(topic))
        ax.set_title("epoch " + str(epoch))
        ax.set_ylabel("distribution")
        ax.set_xlabel("interaction step")
        ax.legend()

    plt.savefig("topic_distribution_over_time_EcoAgent.png")

def run_experiment_Ecoagent(env_config, user_value_model_config, creator_value_model_config,
      actor_model_config, exp_config, num_interaction):

    env = environment.create_gym_environment(env_config)
    user_value_model = value_model.UserValueModel(**user_value_model_config)
    creator_value_model = value_model.CreatorValueModel(
        **creator_value_model_config)
    actor_model = agent.PolicyGradientAgent(
        user_model=user_value_model,
        creator_model=creator_value_model,
        **actor_model_config)

    runner_ = runner.Runner(env, actor_model, num_interaction)

    num_users = []  # Shape (sum(run_trajectory_length)).
    num_creators = []  # Shape (sum(run_trajectory_length)).
    num_documents = []  # Shape (sum(run_trajectory_length)).
    viable_creators = []
    topic_distribution = []
    topic_sum = []
    selected_probs = []  # Shape (sum(run_trajectory_length)).
    policy_probs = []  # Shape (sum(run_trajectory_length), num_candidates).
          # Collect training data.
    #viable_creators.append(runner_.env.creators)
    print("num_interaction", num_interaction )
    print("at the beginning of the simulation we have ", runner_.env.num_creators, "creators")

    (user_dict, creator_dict, preprocessed_user_candidates, _, probs, _,
     _, topic_distribution, topic_sum, viable_creators, viable_users ) = runner_.run()
    #viable_creators.append(runner_.env.creators)
    #print(runner_.env.num_creators)
    """num_users.append(runner_.env.num_users)

     #I think this is not needed
    num_creators.append(runner_.env.num_creators)      #num viable creator at the end of the simulation
    num_documents.append(runner_.env.num_documents)
    selected_probs.extend(probs['selected_probs'])
    policy_probs.extend(probs['policy_probs'])"""
    print("at the end of the simulation we have ", runner_.env.num_creators, "creators")
    return topic_distribution, topic_sum, viable_creators, viable_users

def gini(arr):
    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_

#to write better, using cycle
def lorenz_curve(start_satisfaction_arr, end_satisfaction_arr, ax  ):
    ## first sort
    sorted_arr = start_satisfaction_arr.copy()
    sorted_arr.sort()
    print(sorted_arr)
    X_lorenz = sorted_arr.cumsum() / sorted_arr.sum()
    print(X_lorenz)
    X_lorenz = np.insert(X_lorenz, 0, 0)
    #X_lorenz[0], X_lorenz[-1]
    #fig = plt.figure(figsize=[6,6])
    #ax = fig.add_subplot(3,4, 1)
    ## scatter plot of Lorenz curve
    ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,
               marker='x', color='red', s=100,  label = "satisfaction at time 0")
    ## line plot of equality
    ax.plot([0,1], [0,1], color='k', label = "line of equality")

    sorted_arr = end_satisfaction_arr.copy()
    sorted_arr.sort()
    print(sorted_arr)
    X_lorenz = sorted_arr.cumsum() / sorted_arr.sum()
    print(X_lorenz)
    X_lorenz = np.insert(X_lorenz, 0, 0)
    #X_lorenz[0], X_lorenz[-1]
    #fig = plt.figure(figsize=[6,6])
    #ax = fig.add_subplot(3,4, 2)
    ## scatter plot of Lorenz curve
    ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,
               marker='x', color='darkgreen', s=100, label = "satisfaction at time 50")
    ax.legend()
    ax.set_ylabel("comulative % of satisfaction")
    ax.set_xlabel("comulative % of content providers")
    ## line plot of equality
    #ax.plot([0,1], [0,1], color='k')
    #fig.savefig('distributionSatisfaction.png', dpi=fig.dpi)


def plot_viable_creators(agent, viable_creators, experiment, num_interaction, fig, fig7,fig8  ):

    #colors = ['red','green','blue','purple', 'cyan', 'yellow', 'black', 'magenta', 'orange', 'pink']
    n_cp = viable_creators.groupby(["time"]).count()["creators"]
    minValue = np.amin(n_cp)
    maxValue = np.amax(n_cp)
    print("experiment", experiment)
    ax = fig.add_subplot(3,4, experiment)
    x = np.arange(num_interaction)
    #print(viable_creators.groupby(["time"]))
    #print(viable_creators.groupby(["time"]).count())
    ax.plot(x, np.array(n_cp))
    ax.set_title(agent + "number of interaction: " + str(num_interaction))
    ax.set_ylim(0, maxValue)

    ax.set_ylabel("viable creators")
    ax.set_xlabel("interaction step")
    ax.legend()

    get_mean_satisfaction_t = np.array([])
    get_median_satisfaction_t =pd.DataFrame()
    for i in range(num_interaction):
        cp = viable_creators.loc[viable_creators['time'] == i]
        if i == 0 or (i == num_interaction/2) or i == num_interaction -1 :
            for cr in cp["creators"]:
                print(cr.create_observation()["creator_id"])
        cp = cp["creator_satisfaction"]
        x = 10 - len(cp)
        d = pd.Series(0, index=np.arange(x))
        cp = cp.append(d, ignore_index=True)
        get_mean_satisfaction_t = np.append(get_mean_satisfaction_t, cp.mean(axis = 0))
        if i == 0 or (i == num_interaction/2) or i == num_interaction -1 :
            get_median_satisfaction_t = pd.concat([get_median_satisfaction_t, cp.rename(i) ], axis = 1)

    ax = fig.add_subplot(3,4, experiment + 1 )
    ax.plot(np.arange(num_interaction), get_mean_satisfaction_t)
    ax.set_title(agent + "number of interaction: " + str(num_interaction))
    ax.set_ylabel("mean satisfaction")
    ax.set_xlabel("step")

    ax = fig.add_subplot(3,4, experiment + 2)
    ax.boxplot(get_median_satisfaction_t, showfliers=False)

    get_median_satisfaction_t.boxplot()
    print("median: ", get_median_satisfaction_t)

    ax.set_title(agent + "number of interaction: " + str(num_interaction))
    ax.set_ylabel("median satisfaction")
    ax.set_xlabel("step")

    #measure satisfaction distribution inequality

    cp_start = viable_creators.loc[viable_creators['time'] == 0]
    cp_start = cp_start["creator_satisfaction"]
    cp_end = viable_creators.loc[viable_creators['time'] == num_interaction - 1]
    cp_end = cp_end["creator_satisfaction"]
    x = 10 - len(cp_start)
    if x < 10:
        d = pd.Series(0, index=np.arange(x))
        cp_start = cp_start.append(d, ignore_index=True)
        #cp to numpy array

    x = 10 - len(cp_end)
    if x < 10:
        d = pd.Series(0, index=np.arange(x))
        cp_end = cp_end.append(d, ignore_index=True)

    print("gini coefficient at time 0", gini(cp_start.to_numpy())  )
    #lorenz_curve(cp_start.to_numpy())

    print("gini coefficient at time ",num_interaction, gini(cp_end.to_numpy())  )
    ax = fig.add_subplot(3,4, experiment + 3 )

    lorenz_curve(cp_start.to_numpy(), cp_end.to_numpy(), ax)


def plot_topic_distribution(agent, topic_distribution, experiment, num_interaction, fig):

    colors = ['red','green','blue','purple', 'cyan', 'yellow', 'black', 'magenta', 'orange', 'pink']

    minValue = np.amin(topic_distribution)
    maxValue = np.amax(topic_distribution)

    ax = fig.add_subplot(3,4, experiment)
    x = np.arange(num_interaction)
    for topic in range(10):
      #print(topic)
      ax.plot(x, np.array(topic_distribution)[:,topic], label = str(topic), color= colors[topic])
      ax.set_title(agent + "number of interaction: " + str(num_interaction))
      ax.set_ylim(minValue, maxValue)
      ax.set_ylabel("distribution")
      ax.set_xlabel("interaction step")
      ax.legend()

def plot_topic_characteristic_per_timestamp(agent, topic_sum, timestamps, experiment, num_interaction, subplot ):

    #ax = fig.add_subplot(3,4, experiment)
    ax = subplot
    x = np.arange(10)
    colors = ['red','green','blue','purple', 'cyan', 'yellow', 'black', 'magenta', 'orange', 'pink']
    starting_point = np.array(topic_sum)[0]
    ending_point = np.array(topic_sum)[-1]

    width = 0.2
    w = - 0.2
    rects = []

    rects.append(ax.bar(x - 0.4, starting_point, width, label = "starting_point", color= "red") )    #first iteraction array

    for t in timestamps:
        rects.append(ax.bar(x + w , np.array(topic_sum)[t], width, label = t, color= colors[timestamps.index(t)]) )   #last iteraction array
        w += 0.2

    rects.append(ax.bar(x + w , ending_point, width, label = "ending_point", color= "yellow") )    #first iteraction array

    ax.set_title(agent + "number of interaction: " + str(num_interaction))
    ax.set_ylabel("topic count")
    ax.set_xlabel("topic")
    ax.legend()
    print(len(rects))

    for i in range(len(rects)):
        ax.bar_label(rects[i], padding=3)

def get_document_count_cp(agent, viable_creators, experiment, num_interaction, fig):
    #Document and creator are completely splitted entity, I need to link them some way
    x = np.arange(10)
    sum = 0

    document_list = pd.DataFrame()
    document_count = pd.DataFrame()

    for cr in viable_creators['creators']:
      topic_per_cp = []
      missing_elem = []
      document_sum_cp = pd.DataFrame()
      #print(viable_creators['documents'])
      cr_doc = np.array(viable_creators.loc[viable_creators['creators'] == cr]['documents'])
      cr_doc = [item for sublist in cr_doc for item in sublist]

      #print("cr_doc ",cr.create_observation()['creator_id'], len(cr_doc))  #flattening the list
      for doc in cr_doc:
          #print(doc)

          doc = pd.DataFrame.from_dict([doc.create_observation_nominal()])
          document_list = document_list.append(doc)                     #contains the list of all the documents
          document_sum_cp = document_sum_cp.append(doc)                 #contains the list of document for the given content provider
          topic = int(doc['topic'].to_string(index=False))

          if topic not in topic_per_cp:
              #getting founded topics
              topic_per_cp.append(topic)
          missing_elem = [ele for ele in range(10) if ele not in topic_per_cp]

      #Reset the index of the DataFrame, and use the default one instead. If the DataFrame has a MultiIndex, this method can remove one or more levels.
      document_count = document_count.append(document_sum_cp.groupby(["creator_id","topic"]).count().reset_index())
      #print(len(cr.documents),  document_list)
      #change doc_id with count
      for elem in missing_elem:
          fake_doc = {
              'doc_id': int(0),
              'topic': elem,
              'creator_id': int(doc['creator_id']),
          }
          fake_doc = pd.DataFrame.from_dict([fake_doc])
          document_count = document_count.append(fake_doc)

    document_count = document_count.reset_index(drop=True)
    document_count = document_count.rename(columns={"doc_id": "topic_count"})
    document_count["status"] = "stayed"

    present_cp = np.unique(document_count["creator_id"])
    gone_cp = [ele for ele in range(10) if ele not in present_cp]
    #doc insert column cp stay/cp left
    for elem in gone_cp:
        for i in range(10):
            fake_doc = {
                'topic_count': int(0),
                'topic': i,
                'creator_id': elem,
                'status': 'gone',
            }
            fake_doc = pd.DataFrame.from_dict([fake_doc])
            document_count = document_count.append(fake_doc)


    char = pd.DataFrame()

    ind = {'Topic': ['topic1', 'topic2', 'topic3', 'topic4', 'topic6', 'topic6', 'topic7', 'topic8', 'topic9', 'topic10']}
    char = pd.DataFrame(ind).set_index('Topic')
    #####char =            #contentProvider1            ...            contentProvider
                #topic1     n doc with topic per cp
                # ...
                # topic10

    for i in range(10):
      char[i] = np.array(document_count.loc[document_count["creator_id"] == i].sort_values('topic')['topic_count']).reshape(10, 1)

    char = char.rename(columns={0: "cp0", 1: "cp1", 2: "cp2", 3: "cp3", 4: "cp4", 5: "cp5", 6: "cp6", 7: "cp7", 8: "cp8", 9: "cp9"})

    return char        #return document count matrix per cp/topic

def plot_topic_characteristic_cp_timestamp(agent, viable_creators, timestamps, experiment, num_interaction, fig):
    half = int(num_interaction/2)
    char_start = get_document_count_cp(agent, viable_creators.loc[viable_creators['time'] == 0], experiment, num_interaction, fig)

    char_end = get_document_count_cp(agent, viable_creators.loc[viable_creators['time'] == num_interaction - 1], experiment, num_interaction, fig)
    colors = ['red','green','blue','purple', 'cyan', 'yellow', 'black', 'magenta', 'orange', 'pink']
    maxValue = np.maximum(np.amax(char_start.max(axis=1)), np.amax(char_end.max(axis=1)))

    print("max: ", maxValue)                    #return the maximum value for each row

    for i in range(10):
        ax = fig.add_subplot(5,2, i+1)
        #fig.tight_layout(pad = 20)
        x = np.arange(10)
        starting_point = char_start.iloc[i]
        ending_point = char_end.iloc[i]

        width = 0.2
        w = - 0.2
        rects = []

        rects.append(ax.bar(x - 0.4, starting_point, width, label = "starting_point", color= "red") )    #first iteraction array

        for t in timestamps:
            #print("eccomi", viable_creators.loc[viable_creators['time'] == t])
            rects.append(ax.bar(x + w , get_document_count_cp(agent, viable_creators.loc[viable_creators['time'] == t], experiment, num_interaction, fig ).iloc[i], width, label = t, color= colors[timestamps.index(t)]) )   #last iteraction array
            w += 0.2

        rects.append(ax.bar(x + w , ending_point, width, label = "ending_point", color= "yellow") )    #first iteraction array

        ax.set_ylim(0, maxValue+2)
        ax.set_title("Document count per topic/cp: " + str(i))
        ax.set_ylabel("Topic count")
        ax.set_xlabel("Cp")
        ax.legend()

        for i in range(len(rects)):
            ax.bar_label(rects[i], padding=3)

def rescaled_topic_distribution_plot(topic_distribution_EcoAgent, topic_distribution_RandomAgent, experiment, num_interaction, fig):

    colors = ['red','green','blue','purple', 'cyan', 'yellow', 'black', 'magenta', 'orange', 'pink']

    minValueEA = np.amin(topic_distribution_EcoAgent)
    maxValueEA = np.amax(topic_distribution_EcoAgent)
    minValueRA = np.amin(topic_distribution_RandomAgent)
    maxValueRA = np.amax(topic_distribution_RandomAgent)


    ax = fig.add_subplot(3,4, experiment)
    x = np.arange(num_interaction)

    for topic in range(10):
      ax.plot(x, np.array(topic_distribution_EcoAgent)[:,topic], label = str(topic), color= colors[topic])
      ax.set_title("Rescaled Ecoagent number of interaction: " + str(num_interaction))
      ax.set_ylim(minValueRA, maxValueRA)
      ax.set_ylabel("Distribution")
      ax.set_xlabel("Interaction step")
      ax.legend()

    ax = fig.add_subplot(3,4, experiment+2)
    x = np.arange(num_interaction)

    for topic in range(10):
      ax.plot(x, np.array(topic_distribution_RandomAgent)[:,topic], label = str(topic), color= colors[topic])
      ax.set_title(" Rescaled RandomAgent number of interaction: " + str(num_interaction))
      ax.set_ylim(minValueEA, maxValueEA)
      ax.set_ylabel("Distribution")
      ax.set_xlabel("Interaction step")
      ax.legend()

def plot_cr_topic_preference_over_time(agent, viable_creators, actor, timestamps, experiment, num_interaction, fig):
    viable_creators = viable_creators.reset_index(drop = True)
    colors = ['red','green','blue','purple', 'cyan', 'yellow', 'black', 'magenta', 'orange', 'pink']




    #print(len(viable_creators[actor].unique()))
    n = 0
    for cr in viable_creators[actor].unique():
        n += 1
        ax = fig.add_subplot(5,2, n)
        taking_time = len(np.vstack(viable_creators.loc[viable_creators[actor] == cr]['time']))
        #print(viable_creators.loc[viable_creators[actor] == cr])
        #print("time", taking_time)
        x = np.arange(taking_time)
        pp = np.vstack(viable_creators.loc[viable_creators[actor] == cr]['topic_preference'])
        #print(actor , pp)
        #print(viable_creators.groupby(["time"]).count())
        #missing_at_time_t = [ele for ele in range(num_interaction) if ele not in taking_time]
        #print(missing_at_time_t)
        """for i in range(num_interaction - taking_time):
            pp = np.vstack([pp, 1])

            print(pp)
            print("jakgnjv")

        print(pp, pp.shape)"""

        """for elem in gone_cp:
            for i in range(10):
                fake_doc = {
                    'topic_count': int(0),
                    'topic': i,
                    'creator_id': elem,
                    'status': 'gone',
                }
                fake_doc = pd.DataFrame.from_dict([fake_doc])
                document_count = document_count.append(fake_doc)"""

        #print(type(pp), pp[:,0])
        for topic in range(10):
        #  print("tt",topic)
          ax.plot(x, np.array(pp)[:,topic], label = str(topic), color= colors[topic])
         # print("tt", topic)
          if(actor == "creators"):
            ax.set_title("Ecoagent topic preference of " + actor + ": " + str(cr.create_observation()["creator_id"]))
          else:
            ax.set_title("Ecoagent topic preference of " + actor + ": " + str(cr.get_user_id()))
        #print(num_interaction)
        ax.set_xlim(0, num_interaction)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Distribution")
        ax.set_xlabel("Interaction step")
        ax.legend()
    #print("n", n )
    fig.show()
    #fare la stessa cosa per gli utenti, ma essendo gli utenti 50 come li prendo? modifico mettendo 10 utenti?


### adding RANDOM AGENT
def learn_RandomAgent(env_config, user_value_model_config, creator_value_model_config,
          exp_config):
  """Train and test user_value_model and creator_value_model with random agent."""

  env = environment.create_gym_environment(env_config)
  agent_ = agent.RandomAgent(env_config['slate_size'])
  runner_ = runner.Runner(env, agent_, exp_config['nsteps'])

  train_summary_dir = os.path.join(FLAGS.logdir, 'train_random/')
  if( not os.path.isdir(train_summary_dir) ):
      os.makedirs(train_summary_dir)

  log_path = os.path.join(FLAGS.logdir, 'log')

  user_value_model = value_model.UserValueModel(**user_value_model_config)
  creator_value_model = value_model.CreatorValueModel(
      **creator_value_model_config)

  experience_replay = data_utils.ExperienceReplay(exp_config['nsteps'],
                                                  env_config['topic_dim'],
                                                  env_config['num_candidates'],
                                                  exp_config['user_gamma'],
                                                  exp_config['creator_gamma'])

  train_summary_writer = tf.summary.create_file_writer(train_summary_dir)
  fig = plt.figure(figsize=(30,15))

  # Train, save.
  for epoch in range(1, 1 + exp_config['epochs']):
    print("epochs", epoch)

    # Collect training data.
    num_users = []  # Shape (sum(run_trajectory_length)).
    num_creators = []  # Shape (sum(run_trajectory_length)).
    num_documents = []  # Shape (sum(run_trajectory_length)).
    selected_probs = []  # Shape (sum(run_trajectory_length)).
    policy_probs = []  # Shape (sum(run_trajectory_length), num_candidates).
    for _ in range(exp_config['epoch_runs']):
      (user_dict, creator_dict, _, env_record, probs, _, _, topic_distribution) = runner_.run()
      experience_replay.update_experience(
          user_dict, creator_dict, update_actor=False)
      num_users.extend(env_record['user_num'])
      num_creators.extend(env_record['creator_num'])
      num_documents.extend(env_record['document_num'])
      selected_probs.extend(probs['selected_probs'])
      policy_probs.extend(probs['policy_probs'])

    # Update model with training data.
    for batch_data in experience_replay.user_data_generator(
        exp_config['batch_size']):
      user_value_model.train_step(*batch_data)
    for batch_data in experience_replay.creator_data_generator(
        exp_config['batch_size'],
        creator_id_embedding_size=creator_value_model.creator_id_embedding_size
    ):
      creator_value_model.train_step(*batch_data)

    # Write summaries.
    if epoch % exp_config['summary_frequency'] == 0:
      training_utils.save_summary(
          train_summary_writer,
          user_value_model,
          creator_value_model,
          experience_replay,
          num_users,
          num_creators,
          num_documents,
          topic_distribution,
          policy_probs,
          selected_probs,
          epoch=epoch)

      with open(log_path, 'a') as f:
        f.write('Epoch {}, User train Loss: {}, Creator train Loss: {}.'.format(
            epoch, user_value_model.train_loss.result(),
            creator_value_model.train_loss.result()))

    # Reset.
    user_value_model.train_loss.reset_states()
    user_value_model.train_relative_loss.reset_states()
    creator_value_model.train_loss.reset_states()
    creator_value_model.train_relative_loss.reset_states()
    experience_replay.reset()

    if(epoch == 0):
      ax = fig.add_subplot(3,5, 1)

      x = np.arange(20)
      for topic in range(10):
        ax.plot(x, np.array(topic_distribution)[:,topic], label = str(topic))
        ax.set_title("epoch " + str(epoch))
        ax.set_ylabel("Distribution")
        ax.set_xlabel("Interaction step")
        ax.legend()

    if epoch >= exp_config['start_save'] and exp_config[
        'save_frequency'] > 0 and epoch % exp_config['save_frequency'] == 0:
      # Save model.
      user_value_model.save()
      creator_value_model.save()

      ax = fig.add_subplot(3,5, int(epoch/25)+1)

      x = np.arange(20)
      for topic in range(10):
        ax.plot(x, np.array(topic_distribution)[:,topic], label = str(topic))
        ax.set_title("epoch " + str(epoch))
        ax.set_ylabel("Distribution")
        ax.set_xlabel("Interaction step")
        ax.legend()

    plt.savefig("topic_distribution_over_time_randomAgent.png")

def run_experiment_RandomAgent(env_config, user_value_model_config, creator_value_model_config,
          exp_config, num_interaction):
    env = environment.create_gym_environment(env_config)
    agent_ = agent.RandomAgent(env_config['slate_size'])
    runner_ = runner.Runner(env, agent_, num_interaction)


    user_value_model = value_model.UserValueModel(**user_value_model_config)
    creator_value_model = value_model.CreatorValueModel(
        **creator_value_model_config)

    num_users = []  # Shape (sum(run_trajectory_length)).
    num_creators = []  # Shape (sum(run_trajectory_length)).
    num_documents = []  # Shape (sum(run_trajectory_length)).
    topic_distribution = []
    selected_probs = []  # Shape (sum(run_trajectory_length)).
    policy_probs = []  # Shape (sum(run_trajectory_length), num_candidates).
          # Collect training data.

    (user_dict, creator_dict, preprocessed_user_candidates, _, probs, _,
     _, topic_distribution, topic_sum, viable_creators, viable_users) = runner_.run()

    num_users.append(runner_.env.num_users)

    num_creators.append(runner_.env.num_creators)      #num viable creator at the end of the simulation
    num_documents.append(runner_.env.num_documents)
    selected_probs.extend(probs['selected_probs'])
    policy_probs.extend(probs['policy_probs'])

    return topic_distribution, topic_sum, viable_creators

def save_plot_to_pdf(fig1, fig2, pdf):
    pdf.savefig(fig1)
    pdf.savefig(fig2)


#### END OF RANDOM AGENT
def learn_fair(env_config, user_value_model_config, creator_value_model_config, actor_model_config, exp_config):
    print("hello")


def analyze_independent_experiment(env_config, user_value_model_config, creator_value_model_config, actor_model_config, exp_config, interaction_vector,  FLAGS, user_ckpt_save_dir, creator_ckpt_save_dir):
    experiment = 1;
    fig2 = plt.figure(figsize=(50,25))
    fig3 = plt.figure(figsize=(50,25))

    fig6 = plt.figure(figsize=(50,25))
    fig7 = plt.figure(figsize=(50,25))
    fig8 = plt.figure(figsize=(50,25))
    fig3.suptitle("Analysis of 3 Independent Experimemnts with different number of interaction", fontsize=13)

    fig6.suptitle("Viable creators over time", fontsize=13)

    topicOverCP = PdfPages('TopicOverCP.pdf')
    topic_pref = PdfPages('TopicPreference.pdf')


    for n in interaction_vector:                                            #running independent experiment and plotting topic characteristic for each experiment

        topic_distribution_EcoAgent, topic_sum_EcoAgent, viable_creators_EcoAgent, viable_users = run_experiment_Ecoagent(env_config, user_value_model_config, creator_value_model_config, actor_model_config, exp_config, n)
        topic_distribution_RandomAgent, topic_sum_RandomAgent, viable_creators_RandomAgent = run_experiment_RandomAgent(env_config, user_value_model_config, creator_value_model_config, exp_config, n)
        plot_topic_distribution("EcoAgent ", topic_distribution_EcoAgent, experiment, n, fig2)
        plot_topic_distribution("RandomAgent ", topic_distribution_RandomAgent, experiment + 2 , n, fig2)
        rescaled_topic_distribution_plot(topic_distribution_EcoAgent, topic_distribution_RandomAgent, experiment + 1, n, fig2)
        fig2.savefig("topic_distribution_over_time.pdf")

        plot_topic_characteristic_per_timestamp("EcoAgent ", topic_sum_EcoAgent, [], experiment + 1 , n,  fig3.add_subplot(3,4,  experiment + 1))
        plot_topic_characteristic_per_timestamp("RandomAgent ", topic_sum_RandomAgent, [], experiment + 2 , n,  fig3.add_subplot(3,4,  experiment + 2))

        fig = plt.figure(figsize=(60,50))
        figx = plt.figure(figsize=(60,50))
        fig.suptitle("EcoAgent Num. interaction " + str(n), fontsize=13)
        figx.suptitle("RandomAgent Num. interaction " + str(n), fontsize=13)
        plot_topic_characteristic_cp_timestamp("EcoAgent ", viable_creators_EcoAgent, [], experiment + 1 , n, fig)
        plot_topic_characteristic_cp_timestamp("RandomAgent ", viable_creators_RandomAgent, [], experiment + 1 , n, figx)
        fig4 = plt.figure(figsize=(50,25))
        fig5 = plt.figure(figsize=(50,25))
        fig4.suptitle("Analysis of content providers' topic_preference_over_time", fontsize=13)
        fig5.suptitle("Analysis of users' topic preference over time", fontsize=13)
        plot_cr_topic_preference_over_time("EcoAgent ", viable_creators_EcoAgent, 'creators', [], experiment + 1 , n, fig4)
        plot_cr_topic_preference_over_time("EcoAgent ", viable_users,  'users' , [], experiment + 1 , n, fig5)
        save_plot_to_pdf(fig, figx, topicOverCP)
        save_plot_to_pdf(fig4, fig5, topic_pref)
        #save_plot_to_pdf2.savefig(figx)
        #plt.show()

        plot_viable_creators("EcoAgent ", viable_creators_EcoAgent, experiment, n, fig6, fig7,fig8 )
        fig6.savefig('viable_creators_EcoAgent.png', dpi=fig.dpi)
        #fig7.savefig('mean.png', dpi=fig.dpi)
        #fig8.savefig('median.png', dpi=fig.dpi)

        experiment = experiment + 4
        user_value_model_config = reset_user_model(env_config,  FLAGS, user_ckpt_save_dir)
        creator_value_model_config = reset_creator_model(env_config, FLAGS, creator_ckpt_save_dir)
        #TODO:  plot_topic_characteristic_per_timestamp AND plot_topic_characteristic_cp_timestamp are extremely similar. Try to create a single function using params

    fig = plt.figure(figsize=(60,50))
    figx = plt.figure(figsize=(60,50))
    figy = plt.figure(figsize=(50,25))

    figy.suptitle("Analysis of Long Term Experimemnt over different timesta mps " + str(n) , fontsize=13)
    #analize in different timestamps the plot for the long term experiment, so something like This
    half = int(interaction_vector[-1]/2)
    print(half)
    plot_topic_characteristic_per_timestamp("EcoAgent ", topic_sum_EcoAgent, [half], 1 , n, figy.add_subplot(1,2, 1))     #50, 100
    plot_topic_characteristic_per_timestamp("RandomAgent ", topic_sum_RandomAgent, [half ], 2 , n, figy.add_subplot(1,2, 2))

    plot_topic_characteristic_cp_timestamp("EcoAgent ", viable_creators_EcoAgent, [half], experiment + 1 , n, fig)
    plot_topic_characteristic_cp_timestamp("RandomAgent ", viable_creators_RandomAgent,[half] , experiment + 1 , n, figx)

    save_plot_to_pdf(fig, figx, topicOverCP)
    topicOverCP.close()

    topicPlot = PdfPages('TopicPlot.pdf')
    save_plot_to_pdf(fig3, figy, topicPlot)
    topicPlot.close()
    topic_pref.close()

def reset_user_model(env_config, FLAGS, user_ckpt_save_dir):
    user_value_model_config = {
        'document_feature_size': env_config['topic_dim'],
        'creator_feature_size': None,
        'user_feature_size': None,
        'input_reward': False,
        'regularization_coeff': None,
        'rnn_type': FLAGS.user_rnn_type,
        'hidden_sizes': [
            int(size) for size in FLAGS.user_hidden_sizes.split(',')
        ],
        'lr': FLAGS.learning_rate,
        'model_path': user_ckpt_save_dir,
    }
    return user_value_model_config

def reset_creator_model(env_config, FLAGS, creator_ckpt_save_dir):
    creator_value_model_config ={
        'document_feature_size': env_config['topic_dim'],
        'creator_feature_size': 1,
        'regularization_coeff': None,
        'rnn_type': FLAGS.creator_rnn_type,
        'hidden_sizes': [
            int(size) for size in FLAGS.creator_hidden_sizes.split(',')
        ],
        'lr': FLAGS.learning_rate,
        'model_path': creator_ckpt_save_dir,
        'num_creators': env_config['num_creators'],
        'creator_id_embedding_size': FLAGS.creator_id_embedding_size,
        'trajectory_length': FLAGS.nsteps,
    }

    return creator_value_model_config

def main(unused_argv):

  num_users, num_creators = 50, 10
  half_creators = num_creators // 2
  env_config = {
      # Hyperparameters for environment.
      'resample_documents':
          True,
      'topic_dim':
          10,
      'choice_features':
          dict(),
      'sampling_space':
          'unit ball',
      'num_candidates':
          10,
      # Hyperparameters for users.
      'num_users':
          num_users,
      'user_quality_sensitivity': [0.3] * num_users,
      'user_topic_influence': [0.2] * num_users,
      'observation_noise_std': [0.05] * num_users,
      'user_initial_satisfaction': [10.0] * num_users,
      'user_satisfaction_decay': [1.0] * num_users,
      'user_viability_threshold': [0.0] * num_users,
      'user_model_seed':
          list(range(num_users)),
      'slate_size':
          4,
      # Hyperparameters for creators and documents.
      'num_creators':
          num_creators,
      'creator_initial_satisfaction': [5.0] * num_creators,
      'creator_viability_threshold': [0.0] * num_creators,
      'creator_no_recommendation_penalty': [1.0] * num_creators,
      'creator_new_document_margin': [20.0] * num_creators,
      'creator_recommendation_reward':
          [FLAGS.small_recommendation_reward] * half_creators +
          [FLAGS.large_recommendation_reward] * half_creators,
      'creator_user_click_reward': [0.1] * num_creators,
      'creator_satisfaction_decay': [1.0] * num_creators,
      'doc_quality_std': [0.1] * num_creators,
      'doc_quality_mean_bound': [0.8] * num_creators,
      'creator_initial_num_docs': [20] * num_creators,
      'creator_is_saturation': [False] * num_creators,
      'creator_topic_influence': [0.2] * num_creators,
      'copy_varied_property':
          FLAGS.copy_varied_property,
  }

  exp_config = {
      'nsteps': FLAGS.nsteps,
      'user_gamma': FLAGS.user_gamma,
      'creator_gamma': FLAGS.creator_gamma,
      'epochs': FLAGS.epochs,
      'epoch_runs': FLAGS.epoch_runs,
      'epoch_trains': FLAGS.epoch_trains,
      'start_save': FLAGS.start_save,
      'save_frequency': FLAGS.save_frequency,
      'batch_size': FLAGS.batch_size,
      'summary_frequency': FLAGS.summary_frequency,
  }
  #ckpt_save_dir = os.path.join(FLAGS.logdir, 'ckpt/')
  ckpt_save_dir = os.path.join(FLAGS.logdir, 'ckpt_fair/')
  user_ckpt_save_dir = os.path.join(ckpt_save_dir, 'user')
  creator_ckpt_save_dir = os.path.join(ckpt_save_dir, 'creator')
  actor_ckpt_save_dir = os.path.join(ckpt_save_dir, 'actor')
  if( not os.path.isdir(ckpt_save_dir) ):
      #shutil.rmtree(ckpt_save_dir)
      os.makedirs(ckpt_save_dir)
      os.makedirs(user_ckpt_save_dir)
      os.makedirs(creator_ckpt_save_dir)
      os.makedirs(actor_ckpt_save_dir)

  user_value_model_config = reset_user_model(env_config, FLAGS, user_ckpt_save_dir)
  creator_value_model_config = reset_creator_model(env_config, FLAGS, creator_ckpt_save_dir)
  """user_value_model_config = {
      'document_feature_size': env_config['topic_dim'],
      'creator_feature_size': None,
      'user_feature_size': None,
      'input_reward': False,
      'regularization_coeff': None,
      'rnn_type': FLAGS.user_rnn_type,
      'hidden_sizes': [
          int(size) for size in FLAGS.user_hidden_sizes.split(',')
      ],
      'lr': FLAGS.learning_rate,
      'model_path': user_ckpt_save_dir,
  }


  creator_value_model_config = {
      'document_feature_size': env_config['topic_dim'],
      'creator_feature_size': 1,
      'regularization_coeff': None,
      'rnn_type': FLAGS.creator_rnn_type,
      'hidden_sizes': [
          int(size) for size in FLAGS.creator_hidden_sizes.split(',')
      ],
      'lr': FLAGS.learning_rate,
      'model_path': creator_ckpt_save_dir,
      'num_creators': env_config['num_creators'],
      'creator_id_embedding_size': FLAGS.creator_id_embedding_size,
      'trajectory_length': exp_config['nsteps']
  }"""

  actor_model_config = {
      'slate_size': env_config['slate_size'],
      'user_embedding_size': user_value_model_config['hidden_sizes'][0],
      'document_embedding_size': env_config['topic_dim'],
      'creator_embedding_size': creator_value_model_config['hidden_sizes'][0],
      'hidden_sizes': [
          int(size) for size in FLAGS.actor_hidden_sizes.split(',')
      ],
      'weight_size': FLAGS.actor_weight_size,
      'lr': FLAGS.learning_rate,
      'entropy_coeff': FLAGS.actor_entropy_coeff,
      'regularization_coeff': 1e-5,
      'loss_denom_decay': FLAGS.loss_denom_decay,
      'social_reward_coeff': FLAGS.social_reward_coeff,
      'model_path': actor_ckpt_save_dir,
  }

  with open(os.path.join(FLAGS.logdir, 'env_config.json'), 'w') as f:
    f.write(json.dumps(env_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'user_value_model_config.json'),
            'w') as f:
    f.write(json.dumps(user_value_model_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'creator_value_model_config.json'),
            'w') as f:
    f.write(json.dumps(creator_value_model_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'actor_model_config.json'), 'w') as f:
    f.write(json.dumps(actor_model_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'exp_config.json'), 'w') as f:
    f.write(json.dumps(exp_config, sort_keys=True, indent=0))

  #training and experiment with EcoAgent
  #learn_fair(env_config, user_value_model_config, creator_value_model_config, actor_model_config, exp_config)

  #learn(env_config, user_value_model_config, creator_value_model_config, actor_model_config, exp_config)
  analyze_independent_experiment(env_config, user_value_model_config, creator_value_model_config, actor_model_config, exp_config, [1500], FLAGS, user_ckpt_save_dir, creator_ckpt_save_dir)

 ### Training and experiment with Random Agent
 #learn_RandomAgent(env_config, user_value_model_config, creator_value_model_config, exp_config)

 # load_trained_model_RandomAgent(env_config, user_value_model_config, creator_value_model_config, exp_config)

if __name__ == '__main__':
  app.run(main)

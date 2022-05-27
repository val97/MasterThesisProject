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

"""Ecosystem gym environment."""

import collections
import itertools
import copy

from absl import flags
from absl import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from recsim import choice_model
from recsim import document
from recsim.simulator import environment
from recsim.simulator import recsim_gym

from recs_ecosystem_creator_rl.environment import creator
from recs_ecosystem_creator_rl.environment import user

FLAGS = flags.FLAGS

NUM_USERS = 10
NUM_CREATORS = 20
ENV_CONFIG = {
    # Hyperparameters for environment.
    'resample_documents': True,
    'topic_dim': 10,
    'choice_features': dict(),
    'sampling_space': 'unit ball',
    'num_candidates': 5,
    # Hyperparameters for users.
    'user_quality_sensitivity': [0.5] * NUM_USERS,
    'user_topic_influence': [0.2] * NUM_USERS,
    'observation_noise_std': [0.1] * NUM_USERS,
    'user_initial_satisfaction': [10] * NUM_USERS,
    'user_satisfaction_decay': [1.0] * NUM_USERS,
    'user_viability_threshold': [0] * NUM_USERS,
    'user_model_seed': list(range(NUM_USERS)),
    'num_users': NUM_USERS,
    'slate_size': 2,
    # Hyperparameters for creators and documents.
    'num_creators': NUM_CREATORS,
    'creator_initial_satisfaction': [10] * NUM_CREATORS,
    'creator_viability_threshold': [0] * NUM_CREATORS,
    'creator_no_recommendation_penalty': [1] * NUM_CREATORS,
    'creator_new_document_margin': [0.5] * NUM_CREATORS,
    'creator_recommendation_reward': [1] * NUM_CREATORS,
    'creator_user_click_reward': [1.0] * NUM_CREATORS,
    'creator_satisfaction_decay': [1.0] * NUM_CREATORS,
    'doc_quality_std': [0.1] * NUM_CREATORS,
    'doc_quality_mean_bound': [0.8] * NUM_CREATORS,
    'creator_initial_num_docs': [2] * NUM_CREATORS,
    'creator_topic_influence': [0.1] * NUM_CREATORS,
    'creator_is_saturation': [True] * NUM_CREATORS,
    'copy_varied_property': None
}


class EcosystemEnvironment(environment.MultiUserEnvironment):
  """Class to represent an ecosystem environment with multiple users and multiple creators.

  Attributes:
    _document_sampler: A sampler to sample documents.
    num_users: Number of viable users on the platform.
    num_creator: Number of viable creators on the platform.
    _slate_size: Number of recommended documents in a slate for a given user.
    user_model: A list of UserModel objects representing each viable user on the
      platform.
    _candidate_set: A dictionary of current document candidates provided to the
      agent to generate recommendation slate for a given user. Key=document.id,
      value=document object.
    _current_documents: Generated from _candidate_set. An ordered dictionary
      with key=document.id, value=document.observable_features.
  """

  def __init__(self, *args, **kwargs):
    super(EcosystemEnvironment, self).__init__(*args, **kwargs)
    if not isinstance(self._document_sampler, creator.DocumentSampler):
      raise TypeError('The document sampler must have type DocumentSampler.')
    logging.info('Multi user environment created.')

  def _do_resample_documents(self):
    """Resample documents without replacement."""
    if self._num_candidates > self.num_documents:
      raise ValueError(
          f'Cannot sample {self._num_candidates} from {self.num_documents} documents.'
      )
    self._candidate_set = document.CandidateSet()
    for doc in self._document_sampler.sample_document(
        size=self._num_candidates):
      self._candidate_set.add_document(doc)

  def reset(self):
    """Resets the environment and return the first observation.

    Returns:
      user_obs: An array of floats representing observations of the user's
        current state.
      doc_obs: An OrderedDict of document observations keyed by document ids.
    """
    self._document_sampler.reset_creator()
    self.user_terminates = dict()
    user_obs = dict()
    for user_model in self.user_model:
      user_model.reset()
      self.user_terminates[user_model.get_user_id()] = False
      user_obs[user_model.get_user_id()] = user_model.create_observation()
    if self._resample_documents:
      self._do_resample_documents()
    self._current_documents = collections.OrderedDict(
        self._candidate_set.create_observation())

    creator_obs = dict()
    for creator_id, creator_model in self._document_sampler.viable_creators.items(
    ):
      creator_obs[creator_id] = creator_model.create_observation()

    return user_obs, creator_obs, self._current_documents


  @property
  def num_users(self):
    return len(self.user_model) - np.sum(list(self.user_terminates.values()))

  @property
  def users(self):
      return self.user_model

  @property
  def num_creators(self):
    return self._document_sampler.num_viable_creators

  @property
  def creators(self):
    return self._document_sampler.get_viable_creators

  @property
  def num_documents(self):
    return self._document_sampler.num_documents

  @property
  def topic_documents(self):
    return self._document_sampler.topic_documents

  @property
  def number_of_topic_documents(self):
    return self._document_sampler.number_of_topic_documents


  def step(self, slates, t, approach):
    """Executes the action, returns next state observation and reward.

    Args:
      slates: A list of slates, where each slate is an integer array of size
        slate_size, where each element is an index into the set of
        current_documents presented.

    Returns:
      user_obs: A list of gym observation representing all users' next state.
      doc_obs: A list of observations of the documents.
      responses: A list of AbstractResponse objects for each item in the slate.
      done: A boolean indicating whether the episode has terminated. An episode
        is terminated whenever there is no user or creator left.
    """


    assert (len(slates) == self.num_users
           ), 'Received unexpected number of slates: expecting %s, got %s' % (
               self._slate_size, len(slates))
    for user_id in slates:
      assert (len(slates[user_id]) <= self._slate_size
             ), 'Slate for user %s is too large : expecting size %s, got %s' % (
                 user_id, self._slate_size, len(slates[user_id]))

    #print("time t", t)
    all_documents = dict()  # Accumulate documents served to each user.
    all_responses = dict(
    )  # Accumulate each user's responses to served documents.
    for user_model in self.user_model:
      if not user_model.is_terminal():
        user_id = user_model.get_user_id()
        # Get the documents associated with the slate.
        doc_ids = list(self._current_documents)  # pytype: disable=attribute-error
        #print("current_documents", len(self._current_documents))
        mapped_slate = [doc_ids[x] for x in slates[user_id]]
        documents = self._candidate_set.get_documents(mapped_slate)
        #print("documents", documents)
        # Acquire user response and update user states.
        responses = user_model.update_state(documents)
        all_documents[user_id] = documents
        all_responses[user_id] = responses


    def flatten(list_):
      return list(itertools.chain(*list_))

    # Update the creators' state: calculating the creator reaction to the current slate
    creator_response, modify_slate = self._document_sampler.update_state(
        flatten(list(all_documents.values())),
        flatten(list(all_responses.values())),t, approach)
    #print("time", t)
    """if(modify_slate):
        print("Here I need to manually modify the slate; this is the current one:  ")
        print(slates)"""
        # get the current slate and modify it a bit


    # Obtain next user state observation.
    self.user_terminates = {
        u_model.get_user_id(): u_model.is_terminal()
        for u_model in self.user_model
    }
    all_user_obs = dict()
    for user_model in self.user_model:
      if not user_model.is_terminal():
        all_user_obs[user_model.get_user_id()] = user_model.create_observation()

    # Obtain next creator state observation.
    all_creator_obs = dict()
    for creator_id, creator_model in self._document_sampler.viable_creators.items(
    ):
      all_creator_obs[creator_id] = creator_model.create_observation()
    #print("all_creator_obs", all_creator_obs)

    # Check if reaches a terminal state and return.
    # Terminal if there is no user or creator on the platform.
    done = self.num_users <= 0 or self.num_creators <= 0

    # Optionally, recreate the candidate set to simulate candidate
    # generators for the next query.
    if self.num_creators > 0:
      if self._resample_documents:
        # Resample the candidate set from document corpus for the next time
        # step recommendation.
        # Candidate set is provided to the agent to select documents that will
        # be recommended to the user.
        self._do_resample_documents()

      # Create observation of candidate set.
      self._current_documents = collections.OrderedDict(
          self._candidate_set.create_observation())
    else:
      self._current_documents = collections.OrderedDict()
    #print("environment resampled doc", self._current_documents)
#get  self._current_documents doc_lenght
    return (all_user_obs, all_creator_obs, self._current_documents,
            all_responses, self.user_terminates, creator_response, done)

  def restore_previous_user_state(self, previous_user_state):
       print("before")
       for u in previous_user_state:
           print(u.create_observation())
       #for u in range(len(self.user_model)):
          # self.user_model[u].restore(previous_user_state[u])
       previous_user_state = copy.deepcopy(self.user_model)
       print("after")
       for u in previous_user_state:
           print(u.create_observation())

  #return the cp with highest satisfaction beetween the one recommended at this timestep
  def get_recommended_cp_with_max_sat(self, cps, recommended):
      cps = cps.sort_values(by = ["creator_previous_satisfaction"], ascending = False)
      #print("sorted", cps )
      for cp in cps["creator_id"]:
          if cp in recommended:
              return cp

  #to be moved in utilities
  def normalize(self, df, column, new_column):
   #min_value, max_value = df[column].min(), df[column].max()
   min_value = 0        #min value sat can have
   max_value =  df[column].max()
   df[new_column] = (df[column] - min_value) / (max_value - min_value)

   return df

  def scale(self, df, column, new_column):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[new_column] = min_max_scaler.fit_transform(df[column].values.astype(float))
    return df


  def rebalance_content_provider_satisfaction(self, previous_cps_state, all_documents_df, creator_response, t ):
       print("trying to rebalance cp satisfaction, in order to obtain fairness")
       creator_df = pd.DataFrame()
       redo_timestep = False
       for cr_id in previous_cps_state:
           if not self._document_sampler.viable_creators[cr_id].viable:
               current_satisfaction = 0
           else:
               current_satisfaction = self._document_sampler.viable_creators[cr_id].satisfaction
           creator_df = creator_df.append(pd.DataFrame( {"creator_id": [cr_id], "creator_satisfaction": [ current_satisfaction ],  "creator_previous_satisfaction": [previous_cps_state[cr_id].satisfaction ], "creator_action": [ creator_response[cr_id][0]  ] }))

       creator_df = self.normalize(creator_df, "creator_previous_satisfaction", "creator_previous_satisfaction_normalized")
       creator_df = self.normalize(creator_df, "creator_previous_satisfaction", "creator_previous_satisfaction_scaled")
       recommended_cp = all_documents_df["creator_id"].unique()
       print("at this timestep", all_documents_df["creator_id"].unique(), "have been recommended" )

       #if cp[satisfaction].max() - y >  alpha where y in cp[satisfaction] and alpha unfairness tollerance
       max_sat = creator_df["creator_previous_satisfaction_normalized"].max()
       print(max_sat, creator_df["creator_previous_satisfaction_normalized"])
       alpha = 0.2 #to be tuned
       substituted_cp = []

       #TODO: all_document_df needs to be update here, otherwise need to get response from user:
                #replace the existing row of cp_max with the new one of cp
                #track down the cp that have been already substituted, just to be sure you don't enter in a loop
       for cp in creator_df["creator_id"]:
         if cp not in recommended_cp:
            print(cp, "not in ", recommended_cp)
            check = creator_df.loc[creator_df["creator_id"] == cp]
            math = round(max_sat - float(check["creator_previous_satisfaction_normalized"]), 1)
            print("math", math)
            if math >  alpha:
                 print("if")
                 #recommend cp instead of the cp with highest satisfaction into the slate
                 docs_cp_max = pd.DataFrame()

                 cp_max = self.get_recommended_cp_with_max_sat(creator_df ,all_documents_df["creator_id"].unique())
                 #gets from the current recommended documents, the documents that has been recommended by the cp with highest satisfaction
                 docs_cp_max = docs_cp_max.append(all_documents_df.loc[all_documents_df["creator_id"]== cp_max])["document_id"].unique()
                 print(" I'm going to substitute ", cp_max, " with cp", cp, docs_cp_max )
                 print(float(creator_df.loc[creator_df["creator_id"] == cp_max]["creator_previous_satisfaction_normalized"]), float(creator_df.loc[creator_df["creator_id"] == cp]["creator_previous_satisfaction_normalized"]))
                 if float(creator_df.loc[creator_df["creator_id"] == cp_max]["creator_previous_satisfaction_normalized"]) > float(creator_df.loc[creator_df["creator_id"] == cp]["creator_previous_satisfaction_normalized"]):
                     print("possible_doc_to_substitute", docs_cp_max[0])

                     get_first_doc_to_replace =  all_documents_df.loc[all_documents_df["document_id"]== docs_cp_max[0]]["doc"].unique()[0]
                     print("tmp_doc_to_substitute", get_first_doc_to_replace)
                     topic = get_first_doc_to_replace.create_observation_nominal()["topic"]
                     tmp = list(self._current_documents)
                     tmp = [int(x) for x in tmp]
                     print("slate before modifying: ", tmp)
                     get_sub_index = tmp.index(get_first_doc_to_replace.create_observation_nominal()["doc_id"])

                     new_doc = self._document_sampler.sample_from_cp( cp, topic, tmp)
                     if new_doc in tmp:
                         new_doc = []
                     print("new_doc", new_doc)
                     if new_doc:
                         substituted_cp.append(cp_max)
                         tmp[get_sub_index] = new_doc.create_observation()["doc_id"]
                         self._candidate_set.add_document(new_doc)
                         #print("newslate: ", get_sub_index, self._candidate_set.get_documents(tmp))
                         update_current_documents = {}
                         for elem in self._candidate_set.get_documents(tmp):
                             update_current_documents[elem.doc_id()] = elem.create_observation()

                         self._current_documents = collections.OrderedDict(update_current_documents)
                         all_documents_df.loc[all_documents_df["document_id"] == docs_cp_max[0], "creator_id"] = cp
                         all_documents_df.loc[all_documents_df["document_id"] == docs_cp_max[0], "doc"] = new_doc
                         all_documents_df.loc[all_documents_df["document_id"] == docs_cp_max[0], "document_id" ] = new_doc.create_observation()["doc_id"]
                         print("slate after modifying: ", list(self._current_documents))
                         redo_timestep = True
                     else:
                         print("he doesn't have eligible doc, so do nothing and let him go")
                         if not self._document_sampler.viable_creators[cp].viable:
                            del self._document_sampler.viable_creators[cp]
                            print("do nothing and let him go")
                            redo_timestep = False

                 else:
                     print("keep as it is ")

            else:
                print("he doesn't need to be recommended, he's satisfaction value is enough  ")
         else:
             print(cp, "in ", recommended_cp)
       print("at the end of the rebalance process, this cp have been recommended", all_documents_df["creator_id"].unique() )
       return redo_timestep



  def keep_content_providers(self, previous_cps_state, all_documents_df, creator_response, t ):
      creator_df = pd.DataFrame()
      redo_timestep = False

      for cr_id in previous_cps_state:
          if not self._document_sampler.viable_creators[cr_id].viable:
              current_satisfaction = 0
          else:
              current_satisfaction = self._document_sampler.viable_creators[cr_id].satisfaction
          creator_df = creator_df.append(pd.DataFrame( {"creator_id": [cr_id], "creator_satisfaction": [ current_satisfaction ],  "creator_previous_satisfaction": [previous_cps_state[cr_id].satisfaction ], "creator_action": [ creator_response[cr_id][0]  ] }))

      print("state at time ",t, "\n ",  creator_df["creator_satisfaction"])
      print("at this timestep", all_documents_df["creator_id"].unique(), "have been recommended" )

      cp_leaving = creator_df.loc[creator_df["creator_action"]=="leave"]
      print("at this timestep", cp_leaving, "wants to leave" )

      docs_cp_max = pd.DataFrame()
      cp = self.get_recommended_cp_with_max_sat(creator_df ,all_documents_df["creator_id"].unique())

      print(cp, all_documents_df)
      #gets from the current recommended documents, the documents that has been recommended by the cp with highest satisfaction
      docs_cp_max = docs_cp_max.append(all_documents_df.loc[all_documents_df["creator_id"]== cp])["document_id"].unique()

      print("possible_doc_to_substitute", docs_cp_max[0])

      get_first_doc_to_replace =  all_documents_df.loc[all_documents_df["document_id"]== docs_cp_max[0]]["doc"].unique()[0]
      print("tmp_doc_to_substitute", get_first_doc_to_replace)
      topic = get_first_doc_to_replace.create_observation_nominal()["topic"]
      tmp = list(self._current_documents)
      tmp = [int(x) for x in tmp]
      print("slate before modifying: ", tmp)
      get_sub_index = tmp.index(get_first_doc_to_replace.create_observation_nominal()["doc_id"])
      #sample new document from cp with min satisfaction
      for cp in cp_leaving["creator_id"]:
          new_doc = self._document_sampler.sample_from_cp( cp, topic, tmp)
          if new_doc:

              tmp[get_sub_index] = new_doc.create_observation()["doc_id"]
              self._candidate_set.add_document(new_doc)
              #print("newslate: ", get_sub_index, self._candidate_set.get_documents(tmp))
              update_current_documents = {}
              for elem in self._candidate_set.get_documents(tmp):
                  #print("elem ", elem.create_observation())
                  update_current_documents[elem.doc_id()] = elem.create_observation()
                  #print("update_current_documents", update_current_documents)
              self._current_documents = collections.OrderedDict(update_current_documents)
              print("slate after modifying: ", list(self._current_documents))
              redo_timestep = True
              return redo_timestep
          else:
              if not self._document_sampler.viable_creators[cp].viable:
                  del self._document_sampler.viable_creators[cp]
                  print("do nothing and let him go")


      print("slate after modifying: ", list(self._current_documents))
      return redo_timestep

  """TODO:
        1.check if any cp is leaving
        2. If so,
            identify the cp who wants to leave
           otherwise 5
            check that he has the minimum satisfaction
            check the cp who has the maximum satisfaction
            remove from the slate the document recommended from the cp with max satisfaction and insert a new document from the cp with min satisfaction
        3.manipulate the slate in order to let him stay on the platform
        4.check that other react fine to the new slates, return to 1
            #users react fine, but their gini coefficient is still highest
                #find a method to decrease it, perhaps rebalancing?
        5.continue the simulation
  """
  def simulate_step(self, slates, t, approach):
    """Executes the action, returns next state observation and reward.

    Args:
      slates: A list of slates, where each slate is an integer array of size
        slate_size, where each element is an index into the set of
        current_documents presented.

    Returns:
      user_obs: A list of gym observation representing all users' next state.
      doc_obs: A list of observations of the documents.
      responses: A list of AbstractResponse objects for each item in the slate.
      done: A boolean indicating whether the episode has terminated. An episode
        is terminated whenever there is no user or creator left.
    """
    assert (len(slates) == self.num_users
           ), 'Received unexpected number of slates: expecting %s, got %s' % (
               self._slate_size, len(slates))
    for user_id in slates:
      assert (len(slates[user_id]) <= self._slate_size
             ), 'Slate for user %s is too large : expecting size %s, got %s' % (
                 user_id, self._slate_size, len(slates[user_id]))

    all_documents = dict()  # Accumulate documents served to each user.
    all_responses = dict(
    )  # Accumulate each user's responses to served documents.

    previous_user_state = copy.deepcopy(self.user_model)
    previous_cps_state = copy.deepcopy(self._document_sampler.viable_creators)
    redo_timestep =True

    while(redo_timestep):
        print("time t", t)
        #restore previous user  state
        for u in range(len(self.user_model)):
           self.user_model[u].restore(previous_user_state[u])

        for user_model in self.user_model:
          if not user_model.is_terminal():
            user_id = user_model.get_user_id()
            # Get the documents associated with the slate.
            doc_ids = list(self._current_documents)  # pytype: disable=attribute-error

            mapped_slate = [doc_ids[x] for x in slates[user_id]]

            #print("mapped_slate", doc_ids,  mapped_slate)
            documents = self._candidate_set.get_documents(mapped_slate)
            # Acquire user response and update user states.
            responses = user_model.simulate_update_state(documents)
            all_documents[user_id] = documents
            all_responses[user_id] = responses

        all_documents_df = pd.DataFrame()

        for user_model in self.user_model:
            user_id = user_model.get_user_id()
            #print("time", t, "all documents for user_id ",user_id, len(all_documents[user_id]))
            for doc in all_documents[user_id]:
                all_documents_df = all_documents_df.append(pd.DataFrame({"user_id": [user_id], "doc": doc, "document_id": [doc.create_observation()["doc_id"]], "topic": [doc.create_observation_nominal()["topic"]], "creator_id": [doc.create_observation()["creator_id"]]}))

        def flatten(list_):
          return list(itertools.chain(*list_))

        self._document_sampler.restore_creator_state(previous_cps_state)


        # Update the creators' state.
        creator_response, modify_slate = self._document_sampler.update_state(
            flatten(list(all_documents.values())),
            flatten(list(all_responses.values())), t, approach)



        #redo_timestep = self.rebalance_content_provider_satisfaction(previous_cps_state, all_documents_df, creator_response, t  )

        if(modify_slate):
            redo_timestep = self.keep_content_providers(previous_cps_state, all_documents_df, creator_response, t  ) # t is just for debugging reasons
        else:
            redo_timestep = False

    # Obtain next user state observation.
    self.user_terminates = {
        u_model.get_user_id(): u_model.is_terminal()
        for u_model in self.user_model
    }
    all_user_obs = dict()
    for user_model in self.user_model:
      if not user_model.is_terminal():
        all_user_obs[user_model.get_user_id()] = user_model.create_observation()


    # Obtain next creator state observation.
    all_creator_obs = dict()
    for creator_id, creator_model in self._document_sampler.viable_creators.items():
      all_creator_obs[creator_id] = creator_model.create_observation()

    # Check if reaches a terminal state and return.
    # Terminal if there is no user or creator on the platform.
    done = self.num_users <= 0 or self.num_creators <= 0

    # Optionally, recreate the candidate set to simulate candidate
    # generators for the next query.
    if self.num_creators > 0:
      if self._resample_documents:
        # Resample the candidate set from document corpus for the next time
        # step recommendation.
        # Candidate set is provided to the agent to select documents that will
        # be recommended to the user.
        self._do_resample_documents()

      # Create observation of candidate set.
      self._current_documents = collections.OrderedDict(
          self._candidate_set.create_observation())
      #print("sampled documents: ", self._current_documents )
    else:
      self._current_documents = collections.OrderedDict()
    #print("environment resampled doc", self._current_documents)
  #get  self._current_documents doc_lenght
    return (all_user_obs, all_creator_obs, self._current_documents,
            all_responses, self.user_terminates, creator_response, done)

  def simulate_step_rebalance(self, slates, t, approach):
   """Executes the action, returns next state observation and reward.

   Args:
     slates: A list of slates, where each slate is an integer array of size
       slate_size, where each element is an index into the set of
       current_documents presented.

   Returns:
     user_obs: A list of gym observation representing all users' next state.
     doc_obs: A list of observations of the documents.
     responses: A list of AbstractResponse objects for each item in the slate.
     done: A boolean indicating whether the episode has terminated. An episode
       is terminated whenever there is no user or creator left.
   """
   assert (len(slates) == self.num_users
          ), 'Received unexpected number of slates: expecting %s, got %s' % (
              self._slate_size, len(slates))
   for user_id in slates:
     assert (len(slates[user_id]) <= self._slate_size
            ), 'Slate for user %s is too large : expecting size %s, got %s' % (
                user_id, self._slate_size, len(slates[user_id]))

   all_documents = dict()  # Accumulate documents served to each user.
   all_responses = dict(
   )  # Accumulate each user's responses to served documents.

   previous_user_state = copy.deepcopy(self.user_model)
   previous_cps_state = copy.deepcopy(self._document_sampler.viable_creators)
   redo_timestep =True

   while(redo_timestep):
       print("time t", t)
       #restore previous user  state
       for u in range(len(self.user_model)):
          self.user_model[u].restore(previous_user_state[u])

       for user_model in self.user_model:
         if not user_model.is_terminal():
           user_id = user_model.get_user_id()
           # Get the documents associated with the slate.
           doc_ids = list(self._current_documents)  # pytype: disable=attribute-error

           mapped_slate = [doc_ids[x] for x in slates[user_id]]

           #print("mapped_slate", doc_ids,  mapped_slate)
           documents = self._candidate_set.get_documents(mapped_slate)
           # Acquire user response and update user states.
           responses = user_model.simulate_update_state(documents)
           all_documents[user_id] = documents
           all_responses[user_id] = responses

       all_documents_df = pd.DataFrame()

       for user_model in self.user_model:
           user_id = user_model.get_user_id()
           #print("time", t, "all documents for user_id ",user_id, len(all_documents[user_id]))
           for doc in all_documents[user_id]:
               all_documents_df = all_documents_df.append(pd.DataFrame({"user_id": [user_id], "doc": doc, "document_id": [doc.create_observation()["doc_id"]], "topic": [doc.create_observation_nominal()["topic"]], "creator_id": [doc.create_observation()["creator_id"]]}))

       def flatten(list_):
         return list(itertools.chain(*list_))

       self._document_sampler.restore_creator_state(previous_cps_state)


       # Update the creators' state.
       creator_response, modify_slate = self._document_sampler.update_state(
           flatten(list(all_documents.values())),
           flatten(list(all_responses.values())), t, approach)



       redo_timestep = self.rebalance_content_provider_satisfaction(previous_cps_state, all_documents_df, creator_response, t  )

       """if(modify_slate):
           redo_timestep = self.keep_content_providers(previous_cps_state, all_documents_df, creator_response, t  ) # t is just for debugging reasons
       else:
           redo_timestep = False"""

   # Obtain next user state observation.
   self.user_terminates = {
       u_model.get_user_id(): u_model.is_terminal()
       for u_model in self.user_model
   }
   all_user_obs = dict()
   for user_model in self.user_model:
     if not user_model.is_terminal():
       all_user_obs[user_model.get_user_id()] = user_model.create_observation()


   # Obtain next creator state observation.
   all_creator_obs = dict()
   for creator_id, creator_model in self._document_sampler.viable_creators.items():
     all_creator_obs[creator_id] = creator_model.create_observation()

   # Check if reaches a terminal state and return.
   # Terminal if there is no user or creator on the platform.
   done = self.num_users <= 0 or self.num_creators <= 0

   # Optionally, recreate the candidate set to simulate candidate
   # generators for the next query.
   if self.num_creators > 0:
     if self._resample_documents:
       # Resample the candidate set from document corpus for the next time
       # step recommendation.
       # Candidate set is provided to the agent to select documents that will
       # be recommended to the user.
       self._do_resample_documents()

     # Create observation of candidate set.
     self._current_documents = collections.OrderedDict(
         self._candidate_set.create_observation())
     #print("sampled documents: ", self._current_documents )
   else:
     self._current_documents = collections.OrderedDict()
   #print("environment resampled doc", self._current_documents)
 #get  self._current_documents doc_lenght
   return (all_user_obs, all_creator_obs, self._current_documents,
           all_responses, self.user_terminates, creator_response, done)


def aggregate_multi_user_reward(responses):

  def _generate_single_user_reward(responses):
    for response in responses:
      if response['click']:
        return response['reward']
    return -1

  return np.sum([
      _generate_single_user_reward(response)
      for _, response in responses.items()
  ])


def _assert_consistent_env_configs(env_config):
  """Raises ValueError if the env_config values are not consistent."""
  # User hparams should have the length num_users.
  if len(env_config['user_initial_satisfaction']) != env_config['num_users']:
    raise ValueError(
        'Length of `user_initial_satisfaction` should be the same as number of users.'
    )
  if len(env_config['user_satisfaction_decay']) != env_config['num_users']:
    raise ValueError(
        'Length of `user_satisfaction_decay` should be the same as number of users.'
    )
  if len(env_config['user_viability_threshold']) != env_config['num_users']:
    raise ValueError(
        'Length of `user_viability_threshold` should be the same as number of users.'
    )
  if len(env_config['user_quality_sensitivity']) != env_config['num_users']:
    raise ValueError(
        'Length of `user_quality_sensitivity` should be the same as number of users.'
    )
  if len(env_config['user_topic_influence']) != env_config['num_users']:
    raise ValueError(
        'Length of `user_topic_influence` should be the same as number of users.'
    )
  if len(env_config['observation_noise_std']) != env_config['num_users']:
    raise ValueError(
        'Length of `observation_noise_std` should be the same as number of users.'
    )
  if len(env_config['user_model_seed']) != env_config['num_users']:
    raise ValueError(
        'Length of `user_model_seed` should be the same as number of users.')

  # Creator hparams should have the length num_creators.
  if len(
      env_config['creator_initial_satisfaction']) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_initial_satisfaction` should be the same as number of creators.'
    )
  if len(
      env_config['creator_viability_threshold']) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_viability_threshold` should be the same as number of creators.'
    )
  if len(
      env_config['creator_new_document_margin']) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_new_document_margin` should be the same as number of creators.'
    )
  if len(env_config['creator_no_recommendation_penalty']
        ) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_no_recommendation_penalty` should be the same as number of creators.'
    )
  if len(env_config['creator_recommendation_reward']
        ) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_recommendation_reward` should be the same as number of creators.'
    )
  if len(env_config['creator_user_click_reward']) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_user_click_reward` should be the same as number of creators.'
    )
  if len(
      env_config['creator_satisfaction_decay']) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_satisfaction_decay` should be the same as number of creators.'
    )
  if len(env_config['doc_quality_std']) != env_config['num_creators']:
    raise ValueError(
        'Length of `doc_quality_std` should be the same as number of creators.')
  if len(env_config['doc_quality_mean_bound']) != env_config['num_creators']:
    raise ValueError(
        'Length of `doc_quality_mean_bound` should be the same as number of creators.'
    )
  if len(env_config['creator_initial_num_docs']) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_initial_num_docs` should be the same as number of creators.'
    )
  if len(env_config['creator_is_saturation']) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_is_saturation` should be the same as number of creators.'
    )
  if len(env_config['creator_topic_influence']) != env_config['num_creators']:
    raise ValueError(
        'Length of `creator_topic_influence` should be the same as number of creators.'
    )


class EcosystemGymEnv(recsim_gym.RecSimGymEnv):
  """Class to wrap recommender ecosystem to gym.Env."""

  def reset(self):
    user_obs, creator_obs, doc_obs = self._environment.reset()
    return dict(
        user=user_obs,
        creator=creator_obs,
        doc=doc_obs,
        total_doc_number=self.num_documents)

  def step(self, action, t, approach):
    (user_obs, creator_obs, doc_obs, user_response, user_terminate,
     creator_response, done) = self._environment.step(action, t, approach)
    obs = dict(
        user=user_obs,
        creator=creator_obs,
        doc=doc_obs,
        user_terminate=user_terminate,
        total_doc_number=self.num_documents,
        user_response=user_response,
        creator_response=creator_response)

    # Extract rewards from responses.
    reward = self._reward_aggregator(user_response)
    info = self.extract_env_info()

    #logging.info(
#        'Environment steps with aggregated %f reward. There are %d viable users, %d viable creators and %d viable documents on the platform.',
#        reward, self.num_users, self.num_creators, self.num_documents)
    #print(self.topic_documents)

    return obs, reward, done, info

  ##TO be removed eventually
  def simulate_step(self, action, t, approach):
    #print("action", action)
    (user_obs, creator_obs, doc_obs, user_response, user_terminate,
     creator_response, done) = self._environment.simulate_step(action, t, approach)
    obs = dict(
        user=user_obs,
        creator=creator_obs,
        doc=doc_obs,
        user_terminate=user_terminate,
        total_doc_number=self.num_documents,
        user_response=user_response,
        creator_response=creator_response)

    # Extract rewards from responses.
    reward = self._reward_aggregator(user_response)
    info = self.extract_env_info()

    #logging.info(
#        'Environment steps with aggregated %f reward. There are %d viable users, %d viable creators and %d viable documents on the platform.',
#        reward, self.num_users, self.num_creators, self.num_documents)
    #print(self.topic_documents)

    return obs, reward, done, info

  def simulate_step_rebalance(self, action, t, approach):
      #print("action", action)
    (user_obs, creator_obs, doc_obs, user_response, user_terminate,
       creator_response, done) = self._environment.simulate_step_rebalance(action, t, approach)
    obs = dict(
          user=user_obs,
          creator=creator_obs,
          doc=doc_obs,
          user_terminate=user_terminate,
          total_doc_number=self.num_documents,
          user_response=user_response,
          creator_response=creator_response)

      # Extract rewards from responses.
    reward = self._reward_aggregator(user_response)
    info = self.extract_env_info()

    return obs, reward, done, info


  @property
  def num_creators(self):
    return self._environment.num_creators

  @property
  def creators(self):
    return self._environment.creators

  @property
  def num_users(self):
    return self._environment.num_users

  @property
  def users(self):
    return self._environment.users

  @property
  def num_documents(self):
    return self._environment.num_documents

  @property
  def topic_documents(self):
    return self._environment.topic_documents

  @property
  def number_of_topic_documents(self):
    return self._environment.number_of_topic_documents


def create_gym_environment(env_config):
  """Return a RecSimGymEnv."""

  _assert_consistent_env_configs(env_config)

  # Build a user model.
  def _choice_model_ctor():
    # env_config['choice_feature'] decides the mass of no-click option when the
    # user chooses one document from the recommended slate. It is a dictionary
    # with key='no_click_mass' and value=logit of no-click. If empty,
    # the MultinomialLogitChoice Model sets the logit of no-click to be -inf,
    # and thus the user has to click one document from the recommended slate.
    return choice_model.MultinomialLogitChoiceModel(
        env_config['choice_features'])

  user_model = []
  for user_id in range(env_config['num_users']):
    user_sampler = user.UserSampler(
        user_id=user_id,
        user_ctor=user.UserState,
        quality_sensitivity=env_config['user_quality_sensitivity'][user_id],
        topic_influence=env_config['user_topic_influence'][user_id],
        topic_dim=env_config['topic_dim'],
        observation_noise_std=env_config['observation_noise_std'][user_id],
        initial_satisfaction=env_config['user_initial_satisfaction'][user_id],
        satisfaction_decay=env_config['user_satisfaction_decay'][user_id],
        viability_threshold=env_config['user_viability_threshold'][user_id],
        sampling_space=env_config['sampling_space'],
        seed=env_config['user_model_seed'][user_id],
    )
    user_model.append(
        user.UserModel(
            slate_size=env_config['slate_size'],
            user_sampler=user_sampler,
            response_model_ctor=user.ResponseModel,
            choice_model_ctor=_choice_model_ctor,
        ))

  # Build a document sampler.
  document_sampler = creator.DocumentSampler(
      doc_ctor=creator.Document,
      creator_ctor=creator.Creator,
      topic_dim=env_config['topic_dim'],
      num_creators=env_config['num_creators'],
      initial_satisfaction=env_config['creator_initial_satisfaction'],
      viability_threshold=env_config['creator_viability_threshold'],
      new_document_margin=env_config['creator_new_document_margin'],
      no_recommendation_penalty=env_config['creator_no_recommendation_penalty'],
      recommendation_reward=env_config['creator_recommendation_reward'],
      user_click_reward=env_config['creator_user_click_reward'],
      satisfaction_decay=env_config['creator_satisfaction_decay'],
      doc_quality_std=env_config['doc_quality_std'],
      doc_quality_mean_bound=env_config['doc_quality_mean_bound'],
      initial_num_docs=env_config['creator_initial_num_docs'],
      topic_influence=env_config['creator_topic_influence'],
      is_saturation=env_config['creator_is_saturation'],
      sampling_space=env_config['sampling_space'],
      copy_varied_property=env_config['copy_varied_property'])

  # Build a simulation environment.
  env = EcosystemEnvironment(
      user_model=user_model,
      document_sampler=document_sampler,
      num_candidates=env_config['num_candidates'],
      slate_size=env_config['slate_size'],
      resample_documents=env_config['resample_documents'],
  )

  return EcosystemGymEnv(env, aggregate_multi_user_reward)

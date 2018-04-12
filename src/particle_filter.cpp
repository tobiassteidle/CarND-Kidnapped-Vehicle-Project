/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 50;
  
  // initialize a random generator
  default_random_engine gen;
  
  // extract values for better readability
  const double std_x = std[0];
  const double std_y = std[1];
  const double std_theta = std[2];
  
  // create distributions for x, y, theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  for(int i = 0; i < num_particles; i++) {
    // create particle
    Particle particle;
    particle.id = i;
    
    // initialize first position and add noise
    particle.x = x + dist_x(gen);
    particle.y = y + dist_y(gen);
    particle.theta = theta + dist_theta(gen);
    
    // set initial weight
    particle.weight = 1.0;
    
    // add to particle list
    particles.push_back(particle);
    
    // init weights
    weights.push_back(particle.weight);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // initialize a random generator
  default_random_engine gen;
  
  // extract values for better readability
  const double std_x = std_pos[0];
  const double std_y = std_pos[1];
  const double std_theta = std_pos[2];
  
  for(int i = 0; i < num_particles; i++) {
    const double x = particles[i].x;
    const double y = particles[i].y;
    const double theta = particles[i].theta;
    
    // calculate predictions
    double pred_x;
    double pred_y;
    double pred_theta;
    
    if(fabs(yaw_rate) < 0.0001) {
      pred_theta = theta;
      pred_x = x + velocity * cos(pred_theta) * delta_t;
      pred_y = y + velocity * sin(theta) * delta_t;
    } else {
      pred_theta = theta + yaw_rate * delta_t;
      pred_x = x + (velocity / yaw_rate) * (sin(pred_theta) - sin(theta));
      pred_y = y + (velocity / yaw_rate) * (cos(theta) - cos(pred_theta));
    }
   
    // create distributions for x, y, theta
    normal_distribution<double> dist_x(pred_x, std_x);
    normal_distribution<double> dist_y(pred_y, std_y);
    normal_distribution<double> dist_theta(pred_theta, std_theta);
    
    // set new values
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

  for (int i = 0; i < observations.size(); i++) {
    // set minimum distance
    double min_dist = numeric_limits<double>::max();
    
    int closest_landmark_id = -1;
    const double obs_x = observations[i].x;
    const double obs_y = observations[i].y;
    
    // determine closest landmark
    for (int j = 0; j < predicted.size(); j++) {
      int pred_id = predicted[j].id;
      double pred_x = predicted[j].x;
      double pred_y = predicted[j].y;
      
      // get current distance
      const double current_dist = dist(obs_x, obs_y, pred_x, pred_y);
      
      // update closest landmark
      if (current_dist < min_dist) {
        closest_landmark_id = pred_id;
        min_dist = current_dist;
      }
    }
    
    // update observation id to closest landmark id
    observations[i].id = closest_landmark_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  
  // init weight normalizer
  double weight_normalizer = 0.0;
  
  for (int i = 0; i < num_particles; i++) {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    
    // transform vehicle to map coordinates
    vector<LandmarkObs> transformed_observations;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs transformed_obs = transformCoordinates(particles[i], observations[j]);
      transformed_obs.id = j;
      transformed_observations.push_back(transformed_obs);
    }
    
    // filter landmarks in sensor range
    vector<LandmarkObs> predicted_landmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
      if ((fabs((particle_x - current_landmark.x_f)) <= sensor_range) && (fabs((particle_y - current_landmark.y_f)) <= sensor_range)) {
        predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }
    
    // call dataAssociation
    dataAssociation(predicted_landmarks, transformed_observations);
    
    // reset weight
    particles[i].weight = 1.0;
    
    const double sigma_x = std_landmark[0];
    const double sigma_y = std_landmark[1];
    const double sigma_x_2 = pow(sigma_x, 2);
    const double sigma_y_2 = pow(sigma_y, 2);
    const double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
    
    for(int j = 0; j < transformed_observations.size(); j++) {
      const double to_x = transformed_observations[j].x;
      const double to_y = transformed_observations[j].y;
      const double to_id = transformed_observations[j].id;
      double weight = 1.0;
      
      for (int k = 0; k < predicted_landmarks.size(); k++) {
        const double lm_id = predicted_landmarks[k].id;
        const double lm_x = predicted_landmarks[k].x;
        const double lm_y = predicted_landmarks[k].y;
        
        if (to_id == lm_id) {
          weight = normalizer * exp(-1.0 * ((pow((to_x - lm_x), 2) / (2.0 * sigma_x_2)) + (pow((to_y - lm_y), 2) / (2.0 * sigma_y_2))));
          particles[i].weight *= weight;
        }
      }
    }
    
    weight_normalizer += particles[i].weight;
  }
  
  // normalize weights
  for (int i = 0; i < particles.size(); i++) {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  // initialize a random generator
  default_random_engine gen;
  
  // random particle index
  uniform_int_distribution<int> particle_ud(0, num_particles - 1);
  int current_index = particle_ud(gen);
  
  double beta = 0.0;
  
  // determine maximum weight
  double max_weight = *max_element(weights.begin(), weights.end()) * 2.0;
  
  // resampling wheel
  vector<Particle> resampled_particles;
  for(int i = 0; i < particles.size(); i++) {
    uniform_real_distribution<double> unrealdist(0.0, max_weight);
    beta += unrealdist(gen);
    
    while(beta > weights[current_index]) {
      beta -= weights[current_index];
      current_index = (current_index + 1) % num_particles;
    }
    
    resampled_particles.push_back(particles[current_index]);
  }
  
  // replace current by resampled particles
  particles = resampled_particles;
}

LandmarkObs ParticleFilter::transformCoordinates(Particle particle, LandmarkObs obs) {
  LandmarkObs transformed_obs;
  transformed_obs.x = particle.x + (cos(particle.theta) * obs.x) - (sin(particle.theta) * obs.y);
  transformed_obs.y = particle.y + (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y);
  return transformed_obs;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  
  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

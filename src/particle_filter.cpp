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
#include <memory>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    std::default_random_engine gen;

    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++)
    {
        Particle particle = Particle();
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;

        particles.push_back(particle);
    }

    weights.reserve(num_particles);

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    std::default_random_engine gen;

    std::normal_distribution<double> dist_x(0.0, std_pos[0]);
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

    if (std::fabs(yaw_rate) == 0)
    {
        std::transform(particles.begin(), particles.end(), particles.begin(),
                       [&](Particle &p) -> Particle {
                           p.x += std::cos(p.theta) * velocity * delta_t + dist_x(gen);
                           p.y += std::sin(p.theta) * velocity * delta_t + dist_y(gen);
                           return p;
                       });
    }
    else
    {
        std::transform(particles.begin(), particles.end(), particles.begin(),
                       [&](Particle &p) -> Particle {
                           p.x += (velocity / yaw_rate) * (std::sin(p.theta + (yaw_rate * delta_t)) - std::sin(p.theta)) + dist_x(gen);
                           p.y += (velocity / yaw_rate) * (std::cos(p.theta) - std::cos(p.theta + (yaw_rate * delta_t))) + dist_y(gen);
                           p.theta += (yaw_rate * delta_t) + dist_theta(gen);
                           return p;
                       });
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations, Particle &particle)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    for (auto &obs : observations)
    {
        auto nearest_elem_it = std::min_element(predicted.begin(), predicted.end(),
                                                [&obs](const LandmarkObs &a, const LandmarkObs &b) -> bool {
                                                    return dist(obs.x, obs.y, a.x, a.y) < dist(obs.x, obs.y, b.x, b.y);
                                                });
        obs.id = std::distance(predicted.begin(), nearest_elem_it);

        associations.push_back(nearest_elem_it->id);
        sense_x.push_back(obs.x);
        sense_y.push_back(obs.y);
    }
    SetAssociations(particle, associations, sense_x, sense_y);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    double norm_dist_normalizer = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    double std_x_2 = std::pow(std_landmark[0], 2);
    double std_y_2 = std::pow(std_landmark[1], 2);

    double sum_weights = 0;
    weights.clear();

    for (unsigned int i = 0; i < particles.size(); i++)
    {
        // 1. Transform to map coordinates
        std::vector<LandmarkObs> transformed_observations(observations.size());
        unsigned int obs_ctr = 0;
        for (auto &obs : observations)
        {
            LandmarkObs transformed_obs;
            transformed_obs.id = 0;
            transformed_obs.x = particles[i].x + (std::cos(particles[i].theta) * obs.x) - (std::sin(particles[i].theta) * obs.y);
            transformed_obs.y = particles[i].y + (std::sin(particles[i].theta) * obs.x) + (std::cos(particles[i].theta) * obs.y);
            transformed_observations[obs_ctr++] = transformed_obs;
        }

        // 2 Get map landmarks in particle sensor range
        std::vector<LandmarkObs> map_landmarks_in_range;
        for (auto &map_landmark : map_landmarks.landmark_list)
        {
            if (dist(map_landmark.x_f, map_landmark.y_f, particles[i].x, particles[i].y) <= sensor_range)
            {
                LandmarkObs tmp_landmark = {map_landmark.id_i, map_landmark.x_f, map_landmark.y_f};
                map_landmarks_in_range.push_back(tmp_landmark);
            }
        }

        // 3. Associate observations with map landmarks
        dataAssociation(map_landmarks_in_range, transformed_observations, particles[i]);

        // 4. Calculate new weights
        long double particle_prob = 1.0;
        for (auto &trans_obs : transformed_observations)
        {
            double prob = norm_dist_normalizer *
                          std::exp(-0.5 * ((std::pow(trans_obs.x - map_landmarks_in_range[trans_obs.id].x, 2) / std_x_2) +
                                           (std::pow(trans_obs.y - map_landmarks_in_range[trans_obs.id].y, 2) / std_y_2)));

            if (prob > 0)
            {
                particle_prob *= prob;
            }
        }
        particles[i].weight = particle_prob;
        weights.push_back(particle_prob);
        sum_weights += particle_prob;
    }

    // 5. Normalize weights
    if (sum_weights > 0)
    {
        std::transform(particles.begin(), particles.end(), particles.begin(),
                       [&sum_weights](Particle &p) -> Particle { p.weight /= sum_weights; return p; });
        std::transform(weights.begin(), weights.end(), weights.begin(),
                       [&sum_weights](double &d) -> double { d /= sum_weights; return d; });
    }
}

void ParticleFilter::resample()
{
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::default_random_engine gen;
    std::discrete_distribution<int> prob_wheel(weights.begin(), weights.end());

    std::vector<Particle> resampled_particles(particles.size());
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        resampled_particles[i] = particles[prob_wheel(gen)];
    }
    particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                     const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    // #ifdef DEBUG
    //     std::cout << "### Called ParticleFilter::SetAssociations() ###"
    //               << "\n";
    // #endif

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

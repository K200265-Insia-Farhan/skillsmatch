"use client"
import React, { useEffect, useState } from 'react';
import Image from 'next/image';
import CandidateProfileSlider from './candidate-profile-slider';
import avatar from '@/assets/images/candidates/img_01.jpg';
import VideoPopup from '../common/video-popup';
import Skills from './skills';
import WorkExperience from './work-experience';
import CandidateBio from './bio';
import EmailSendForm from '../forms/email-send-form';
import axios from 'axios';
import { useSearchParams } from 'next/navigation';
import { set } from 'react-hook-form';
import { profile } from 'console';

const CandidateDetailsArea = () => {
  const [isVideoOpen, setIsVideoOpen] = useState<boolean>(false);
  const [userDetails, setUserDetails] = useState<any>({});
  const [institute, setInstitute] = useState<any>([]);
  const [workExperience, setWorkExperience] = useState<any>([]);
  const [profilePicture, setProfilePicture] = useState<string | null>(null);
  const [skills, setSkills] = useState<any>([]);
  const [resumePath, setResumePath] = useState<string | null>(null);

  const searchParams = useSearchParams();

  const id = searchParams.get('id');
  
  const handleDownloadCV = async () => {
    try {
      // Get the id from search params
      const token = localStorage.getItem('token');

      if (!id) {
        console.error("Candidate id not found in search params");
        return;
      }

      // Make a GET request to the /getResume API endpoint
      const response = await axios.get(`http://100.25.158.124:5000/api/auth/getResume?id=${id}`,{
        headers: {
          Authorization: `Bearer ${token}`
        }
      });

      const newPath = "http://100.25.158.124:5000" + response.data.filePath;

      if (response.status === 200) {
        // Update the state with the resume path received from the API
        setResumePath(newPath);
      } else {
        console.error("Error retrieving resume:", response.data.message);
      }
    } catch (error) {
      console.error("Error retrieving resume:", error);
    }
  };
  useEffect(() => {
    const getCandidateDetailsUsingId = async () => {
      try {
        const response = await axios.get(`http://100.25.158.124:5000/api/auth/getCandidateDetailsUsingId/${id}`);
        console.log("Candidate details: ", response.data.data);
        if(response.status === 200){
          setUserDetails(response.data.data.candidate);
          setInstitute(response.data.data.institute);
          setWorkExperience(response.data.data.workExperience);
          setSkills(response.data.data.candidate.skills);
        }
        // console.log("Skills: ", skills);

      } catch (error){
        console.error("Error fetching candidate details: ", error);
        console.log("Error fetching candidate details: ", error);
      }
    };

    const fetchProfilePicture = async () => {
      try {
        const response = await axios.get(`http://100.25.158.124:5000/api/auth/getProfilePictureUsingId/${id}`);
        console.log("Response: ", response.data.data.filePath);
        if (response.status === 200) {
          // Construct the full URL based on the relative path
          const fullUrl = `http://100.25.158.124:5000${response.data.data.filePath}`;

          // Update the profile picture state with the full URL
          setProfilePicture(fullUrl);
        }
      } catch (error) {
        console.error("Error fetching profile picture:", error);
      }
    };


    getCandidateDetailsUsingId(), fetchProfilePicture();
  }, [])
  return (
    <>
      <section className="candidates-profile pt-100 lg-pt-70 pb-150 lg-pb-80">
        <div className="container">
          <div className="row">
            <div className="col-xxl-9 col-lg-8">
              <div className="candidates-profile-details me-xxl-5 pe-xxl-4">
                <div className="inner-card border-style mb-65 lg-mb-40">
                  <h3 className="title">Overview</h3>
                  <p>{userDetails.overview}</p>
                </div>
                {/* <h3 className="title">Intro</h3>
                <div className="video-post d-flex align-items-center justify-content-center mt-25 lg-mt-20 mb-75 lg-mb-50">
                  <a onClick={() => setIsVideoOpen(true)} className="fancybox rounded-circle video-icon tran3s text-center cursor-pointer">
                    <i className="bi bi-play"></i>
                  </a>
                </div> */}
                <div className="inner-card border-style mb-75 lg-mb-50">
                  <h3 className="title">Education</h3>
                    <div className="time-line-data position-relative pt-15">
                      {institute.map((educationItem: any, index: number) => (
                        <div className="info position-relative">
                          <div className="numb fw-500 rounded-circle d-flex align-items-center justify-content-center">{index+1}</div>
                          <div className="text_1 fw-500">{educationItem.institute_name}</div>
                          <h4>{educationItem.degree_program}</h4>
                          <p>{educationItem.duration}</p>
                        </div>
                      ))}
                    </div>
                </div>
                <div className="inner-card border-style mb-75 lg-mb-50">
                  <h3 className="title">Skills</h3>
                  {/* skill area */}
                  <Skills skills={skills} />
                  {/* skill area */}
                </div>
                <div className="inner-card border-style mb-60 lg-mb-50">
                  <h3 className="title">Work Experience</h3>
                  {/* WorkExperience */}
                  <WorkExperience workExperienceProp={workExperience} />
                  {/* WorkExperience */}
                </div>
                {/* <h3 className="title">Portfolio</h3> */}
                {/* Candidate Profile Slider */}
                {/* <CandidateProfileSlider /> */}
                {/* Candidate Profile Slider */}
              </div>
            </div>
            <div className="col-xxl-3 col-lg-4">
              <div className="cadidate-profile-sidebar ms-xl-5 ms-xxl-0 md-mt-60">
                <div className="cadidate-bio bg-wrapper bg-color mb-60 md-mb-40">
                  <div className="pt-25">
                    <div className="cadidate-avatar m-auto">
                      {/* {profilePicture && (<img src={profilePicture} alt="avatar" className="lazy-img rounded-circle w-100" />)} */}
                      {profilePicture && (
                        <img src={profilePicture} alt="Profile Picture" className="lazy-img rounded-circle" />
                      )}                      
                    </div>
                  </div>
                  <h3 className="cadidate-name text-center">{userDetails.firstname} {userDetails.lastname}</h3>
                  {/* <div className="text-center pb-25"><a href="#" className="invite-btn fw-500">Invite</a></div> */}
                  {/* CandidateBio */}
                  <CandidateBio userDetails={userDetails} workExperience={workExperience} institute={institute} />
                  {/* CandidateBio */}
                  {/* <a href="#" className="btn-ten fw-500 text-white w-100 text-center tran3s mt-15">Download CV</a> */}
                  <a href={resumePath} className="btn-ten fw-500 text-white w-100 text-center tran3s mt-15" onClick={handleDownloadCV} target="_blank">View Resume</a>

                </div>
                {/* <h4 className="sidebar-title">Location</h4>
                <div className="map-area mb-60 md-mb-40">
                  <div className="gmap_canvas h-100 w-100">
                    <iframe className="gmap_iframe h-100 w-100" src="https://maps.google.com/maps?width=600&amp;height=400&amp;hl=en&amp;q=bass hill plaza medical centre&amp;t=&amp;z=12&amp;ie=UTF8&amp;iwloc=B&amp;output=embed"></iframe>
                  </div>
                </div> */}
                {/* <h4 className="sidebar-title">Email James Brower.</h4>
                <div className="email-form bg-wrapper bg-color">
                  <p>Your email address & profile will be shown to the recipient.</p>
                  <EmailSendForm/>
                </div> */}
              </div>
            </div>
          </div>
        </div>
      </section>
      {/* video modal start */}
      <VideoPopup isVideoOpen={isVideoOpen} setIsVideoOpen={setIsVideoOpen} videoId={'-6ZbrfSRWKc'} />
      {/* video modal end */}
    </>
  );
};

export default CandidateDetailsArea;
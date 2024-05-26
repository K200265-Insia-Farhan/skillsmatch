const express = require('express');
const router = express.Router();

const { signupCandidate, 
        loginCandidate,
        createCandidateEducationDetails,
        createCandidateWorkExperience,
        getCandidateDetails, 
        updateCandidateDetails,
        updateCandidateEducationDetails,
        updateCandidateWorkExperience,
        uploadProfilePicture,
        getProfilePicture,
        getAllCandidates,
        getProfilePictureUsingId,
        getCandidateDetailsUsingId,
        getInstituteDetails,
        getWorkExperienceDetails,
        getCandidateDetailsUsingEmail,
        uploadResume,
        getResume,
        applyJob,
        getAppliedJobs,
        getJobDetailsUsingCandidateId,
        saveJob,
        unsaveJob,
        isJobSaved,
        getAllSavedJobs,
        getJobDetailsUsingSavedJobs,
        getCandidateDetailsUsingCandidateIdsArray
    } = require('../controllers/authControllerCandidate');

const{  signupCompanyHR, 
        loginCompanyHR, 
        getCompanyHRDetails, 
        getCompanyDetails, 
        saveCompanyDetails,
        uploadCompanyProfilePicture,
        getCompanyProfilePicture,
        getAllCompanies,
        getCompanyProfilePictureUsingId,
        getCompanyDetailsUsingId,
        getApplicantsUsingJobId,
        saveCandidate,
        unsaveCandidate,
        isCandidateSaved,
        countApplicantsUsingJobId,
        getSavedCandidatesUsingcompanyhrId,
        unsaveCandidateWithoutJobId,
        getCandidateProfilePicture
    } = require('../controllers/authControllerCompanyHR');

const {
    submitJob,
    getJobsbyCompanyHR,
    getAllJobs,
    getJobDetailsUsingId,
    editJob,
    deleteJobUsingId,
    countJobsUsingCompanyHRId,
    getJobsByCompanyHRId,
    getJobsByCareerOfficers
} = require('../controllers/authControllerJobs')

// const jwtMiddlewareCandidate = require('../middleware/jwtMiddlewareCandidate');
// const jwtMiddlewareCompanyHR = require('../middleware/jwtMiddlewareCompanyHR');
// const jwtMiddlewareAllUsers = require('../middleware/jwtMiddlewareAllUsers');

// CANDIDATE CRUD
router.post('/signupCandidate', signupCandidate);
router.post('/loginCandidate', loginCandidate);
router.post('/createCandidateEducationDetails', createCandidateEducationDetails);
router.post('/createCandidateWorkExperience', createCandidateWorkExperience);
router.get('/candidateDetails', getCandidateDetails);
router.put('/updateCandidateDetails', updateCandidateDetails);
router.put('/updateCandidateEducationDetails', updateCandidateEducationDetails);
router.put('/updateCandidateWorkExperience', updateCandidateWorkExperience);
router.post('/uploadProfilePicture', uploadProfilePicture);
router.get('/getProfilePicture', getProfilePicture);
router.get('/getAllCandidates', getAllCandidates);
router.get('/getProfilePictureUsingId/:id', getProfilePictureUsingId);
router.get('/getCandidateDetailsUsingId/:id', getCandidateDetailsUsingId);
router.get('/getInstituteDetails', getInstituteDetails);
router.get('/getWorkExperienceDetails', getWorkExperienceDetails);
router.get('/getCandidateDetailsUsingEmail/:email',getCandidateDetailsUsingEmail);
router.post('/uploadResume', uploadResume);
router.get('/getResume', getResume);
router.get('/getAppliedJobs', getAppliedJobs);
router.get('/getJobDetailsUsingCandidateId/:candidate_id', getJobDetailsUsingCandidateId);
router.post('/saveJob', saveJob);
router.delete('/unsaveJob', unsaveJob);
router.get('/isJobSaved', isJobSaved);
router.get('/getAllSavedJobs', getAllSavedJobs);
router.get('/getJobDetailsUsingSavedJobs/:job_id',getJobDetailsUsingSavedJobs);

// COMPANY HR LOGIN AND SIGNUP ROUTES
router.post('/signupCompanyHR', signupCompanyHR);
router.post('/loginCompanyHR', loginCompanyHR);
router.get('/companyHRDetails',  getCompanyHRDetails);

//COMPANY
router.get('/companyDetails',  getCompanyDetails);
router.put('/saveCompanyDetails',  saveCompanyDetails);
router.post('/uploadCompanyProfilePicture',  uploadCompanyProfilePicture);
router.get('/getCompanyProfilePicture',  getCompanyProfilePicture);
// router.get('/getAllCompanies',  getAllCompanies);
router.get('/getAllCompanies', getAllCompanies);
router.get('/getCompanyProfilePictureUsingId/:id', getCompanyProfilePictureUsingId);
router.get('/getCompanyDetailsUsingId/:id', getCompanyDetailsUsingId);
router.get('/getApplicantsUsingJobId/:job_id',  getApplicantsUsingJobId);
router.post('/saveCandidate',  saveCandidate);
router.post('/unsaveCandidate',  unsaveCandidate);
router.post('/isCandidateSaved',  isCandidateSaved);
router.get('/countApplicantsUsingJobId/:id',  countApplicantsUsingJobId);
router.get('/getSavedCandidatesUsingcompanyhrId/:id',  getSavedCandidatesUsingcompanyhrId);
router.post('/unsaveCandidateWithoutJobId',  unsaveCandidateWithoutJobId);
router.post('/getCandidateProfilePicture',  getCandidateProfilePicture);
router.post('/getCandidateDetailsUsingCandidateIdsArray',getCandidateDetailsUsingCandidateIdsArray);

//JOBS
router.put('/submitJob',  submitJob);
router.get('/getJobsbyCompanyHR',  getJobsbyCompanyHR);
router.get('/getAllJobs',getAllJobs);
// router.get('/getJobDetailsUsingId/:job_id',  getJobDetailsUsingId);
router.get('/getJobDetailsUsingId/:job_id',getJobDetailsUsingId);
router.put('/editJob/:job_id',  editJob);
router.delete('/deleteJobUsingId/:job_id',  deleteJobUsingId);
router.post('/applyJob', applyJob);
router.get('/countJobsUsingCompanyHRId/:id',countJobsUsingCompanyHRId);
router.get('/getJobsByCompanyHRId/:id',getJobsByCompanyHRId);
router.get('/getJobsByCareerOfficers',getJobsByCareerOfficers);

module.exports = router;

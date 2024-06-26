'use client'
import React,{useState} from 'react';
import Wrapper from "@/layouts/wrapper";
import EmployAside from "@/app/components/dashboard/employ/aside";
import SavedCandidateArea from "@/app/components/dashboard/employ/saved-candidate-area";
import AppliedCandidatesArea from "@/app/components/dashboard/employ/applied-candidates-area";

const EmployDashboardSavedCandidatePage = () => {
  const [isOpenSidebar,setIsOpenSidebar] = useState<boolean>(false);
  return (
    <Wrapper>
      <div className="main-page-wrapper">
        {/* aside start */}
        <EmployAside isOpenSidebar={isOpenSidebar} setIsOpenSidebar={setIsOpenSidebar} />
        {/* aside end  */}

        {/* saved candidate area start */}
        <AppliedCandidatesArea setIsOpenSidebar={setIsOpenSidebar} />
        {/* saved candidate area end */}
      </div>
    </Wrapper>
  );
};

export default EmployDashboardSavedCandidatePage;

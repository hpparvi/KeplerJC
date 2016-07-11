module models
  use omp_lib
  implicit none

contains
  subroutine m_transit(depth, center, duration, cadence, npt, model)
    implicit none
    integer, intent(in) :: npt
    real(8), intent(in) :: depth, center, duration
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt) :: model
    real(8) :: hdur
    integer :: i, cstart, cend

    hdur = 0.5d0*duration
    cstart = floor(center - hdur)
    cend = floor(center + hdur)

    do i = 1,npt
       if ((cadence(i) > cstart) .and. (cadence(i) < cend)) then
          model(i) = 1.0d0
       else if (cadence(i) == cstart) then
          model(i) = 1.0d0 - (center - hdur - cstart)
       else if (cadence(i) == cend) then
          model(i) = center + hdur - cend
       else
          model(i) = 0.0d0
       end if
    end do

    model = -depth*model

  end subroutine m_transit

end module models

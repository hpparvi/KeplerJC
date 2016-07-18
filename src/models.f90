module models
  use omp_lib
  implicit none

contains
  subroutine m_transit(depth, center, duration, baseline, cadence, npt, model)
    implicit none
    integer, intent(in) :: npt
    real(8), intent(in) :: depth, center, duration, baseline
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt) :: model
    real(8) :: hdur
    integer :: i, cstart, cend

    hdur = 0.5d0*duration
    cstart = floor(center - hdur)
    cend = floor(center + hdur)

    do concurrent (i = 1:npt)
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

    model = baseline - depth*model
  end subroutine m_transit

  subroutine m_transits(pv, cadence, npt, npar, npv, model)
    implicit none
    integer, intent(in) :: npt, npv, npar
    real(8), intent(in), dimension(npv,npar) :: pv
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt, npv) :: model
    integer :: i, j, cstart(npv), cend(npv)
    real(8) :: hdur(npv)

    hdur = 0.5d0 * pv(:,3)
    cstart = floor(pv(:,2) - hdur)
    cend = floor(pv(:,2) + hdur)

    do concurrent (i=1:npt, j=1:npv)
       if ((cadence(i) > cstart(j)) .and. (cadence(i) < cend(j))) then
          model(i,j) = 1.0d0
       else if (cadence(i) == cstart(j)) then
          model(i,j) = 1.0d0 - (pv(j,2) - hdur(j) - cstart(j))
       else if (cadence(i) == cend(j)) then
          model(i,j) = pv(j,2) + hdur(j) - cend(j)
       else
          model(i,j) = 0.0d0
       end if
       model(i,j) = pv(j,4) - pv(j,1)*model(i,j)
    end do
  end subroutine m_transits

  subroutine m_jump(center, width, amplitude, baseline, cadence, npt, model)
    implicit none
    integer, intent(in) :: npt
    real(8), intent(in) :: center, width, amplitude, baseline
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt) :: model
    real(8) :: hwidth, a,b
    integer :: i, cstart, cend

    hwidth = 0.5d0*width
    cstart = floor(center - hwidth)
    cend = floor(center + hwidth - 1.0d-7)

    do concurrent (i = 1:npt)
       if (cadence(i) < cstart) then
          model(i) = -0.5d0
       else if (cadence(i) > cend) then
          model(i) =  0.5d0
       else
          if (cadence(i) == cstart) then
             a = -hwidth
             b = cadence(i) + 1 - center
          else if (cadence(i) == cend) then
             a = cadence(i) - center
             b = hwidth
          else
             a = cadence(i) - center
             b = cadence(i) + 1 - center
          end if
          model(i) = (b**2 - a**2) / (2.0d0*width*(b-a))
       end if
    end do
    model = baseline + amplitude*model
  end subroutine m_jump

end module models

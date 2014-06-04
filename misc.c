/** @file */

/**
 * @brief Take the integer power of an integer.
 *
 * @param[in] base The base of the exponent.
 * @param[in] power The power of the exponent.
 *
 * @retval answer The value of base**power.
 */
int ipow(int base, int power) {
        int i;
        int answer = 1;
        for (i = 0; i < power; i++) {
                answer *= base;
        }
        return answer;
}

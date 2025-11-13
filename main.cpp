#include <vector>

using namespace std;

class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> preProduct(nums.size(), 1);
        vector<int> postProduct(nums.size(), 1);
        for (int i = 1; i < preProduct.size(); ++i) {
            preProduct[i] = nums[i - 1] * preProduct[i - 1];
        }
        for (int i = nums.size() - 2; i > -1; --i) {
            postProduct[i] = nums[i + 1] * postProduct[i + 1];
        }
        vector<int> ans{};
        ans.reserve(nums.size());
        for (int i = 0; i < nums.size(); ++i) {
            ans.push_back(preProduct[i] * postProduct[i]);
        }
        return ans;
    }
};